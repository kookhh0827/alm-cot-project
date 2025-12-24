import json
import os
import sys
import re # Added regex import for JSON fix
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from main_qwen import QwenModelManager
from validator import OutputValidator

# --- Constants ---
TEARS_V2_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_V2")

# Load Candidates dynamically
CANDIDATES_MAP = {
    "gender": ["female", "male"],
    "age": ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75"],
    "dialect_region": [
        "army brat (moved around)", "new england", "new york city", 
        "north midland", "northern", "south midland", "southern", "western"
    ],
    "ethnicity": [
        "asian", "black or african american", "hispanic or latino", 
        "white or caucasian"
    ]
}
BLOCKLIST = {
    "prefer not to answer", 
    "non-binary / third gender",
    "unknown"
}

def get_candidates(prompt_text):
    text_lower = prompt_text.lower()
    if "gender" in text_lower:
        return CANDIDATES_MAP.get("gender", [])
    elif "age" in text_lower:
        return CANDIDATES_MAP.get("age", [])
    elif "dialect" in text_lower:
        return CANDIDATES_MAP.get("dialect_region", [])
    elif "ethnicity" in text_lower:
        return CANDIDATES_MAP.get("ethnicity", [])
    return []

def flatten_latent(latent_json):
    """
    Flattens nested latent JSON into a single dictionary.
    Example: {"acoustic": {"f0": "low"}} -> {"f0": "low"}
    """
    flat = {}
    for cat, content in latent_json.items():
        if isinstance(content, dict):
            for k, v in content.items():
                flat[k] = v
        else:
            flat[cat] = content
    return flat

def run_part2(split_name: str):
    print(f"=== Part 2: Reasoning Generation for {split_name} (with Validation) ===")
    
    input_path = TEARS_V2_ROOT / f"{split_name}_v3_intermediate.json"
    output_path = TEARS_V2_ROOT / f"{split_name}_v3_final.json"
    
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Please run Part 1 first.")
        return

    with open(input_path, "r") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} items from intermediate file.")
    
    # --- Resume Logic ---
    final_data = []
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                final_data = json.load(f)
            print(f"Resuming: Found {len(final_data)} existing final entries.")
        except Exception as e:
            print(f"Error loading {output_path}: {e}")
            
    processed_keys = set()
    for item in final_data:
        key = (item["audio_path"], item["prompt"])
        processed_keys.add(key)
        
    items_to_process = []
    skipped_count = 0
    for item in data:
        # Check blocklist first
        gt = item.get("ground_truth_answer", "").lower().strip()
        if gt in BLOCKLIST:
            skipped_count += 1
            continue
            
        key = (item["audio_path"], item["prompt"])
        if key not in processed_keys:
            items_to_process.append(item)
            
    print(f"Items remaining to process: {len(items_to_process)} (Skipped {skipped_count} blocklisted/processed)")
    if not items_to_process:
        print("All done!")
        return

    # --- Initialize vLLM ---
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    llm = QwenModelManager.get_model(model_name)
    tokenizer = llm.get_tokenizer()
    
    # --- Load Templates ---
    template_path = Path(__file__).parent / "data_gen_prompt.txt"
    val_template_path = Path(__file__).parent / "data_val_prompt.txt"
    
    if template_path.exists():
        gen_template = template_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError("data_gen_prompt.txt not found")
        
    if val_template_path.exists():
        val_template = val_template_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError("data_val_prompt.txt not found")
    
    sys_prompt = "You are a grounded speaker profiling assistant."
    validator = OutputValidator()
    
    # --- Prepare Prompts ---
    # We will process in batches.
    # Strategy: N=3 generation (Over-generate) -> Filter -> Soft Validate (LLM) -> Pick Best
    
    batch_size = 200 # Lower batch size due to N=3
    num_return_sequences = 3
    
    # Sampling for Generation
    gen_sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=500, # Keep it short as per prompt
        n=num_return_sequences,
        stop=["</RESPONSE>"] # Early stop
    )
    
    # Sampling for Validation (Greedy)
    val_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048 # Increased to avoid JSON truncation
    )

    total_items = len(items_to_process)
    
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_items = items_to_process[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1} ({i}-{batch_end})...")
        
        # 1. Format Generation Prompts
        gen_prompts = []
        candidates_list = []
        
        for item in batch_items:
            latent_str = json.dumps(item["latent_json"], indent=2)
            cands = get_candidates(item['prompt'])
            candidates_list.append(cands)
            cand_str = ", ".join(cands)
            
            user_text = gen_template
            user_text = user_text.replace("{transcript}", item['transcript'])
            user_text = user_text.replace("{latent_json}", latent_str)
            user_text = user_text.replace("{question}", item['prompt'])
            user_text = user_text.replace("{answer}", item['ground_truth_answer'])
            user_text = user_text.replace("{candidate_labels}", cand_str)
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_text},
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            gen_prompts.append(full_prompt)
            
        # 2. Generate (N=3)
        gen_outputs = llm.generate(gen_prompts, gen_sampling_params)
        
        # 3. Validation Logic (Per Item)
        # We need to collect "Best Passing" for each item.
        # If Hard Pass -> Submit to LLM Val -> If Soft Pass -> Success
        
        # For efficiency, we can batch the LLM validation calls too.
        # But that complicates the flow (Gen -> Hard Val -> Gather Soft Val Prompts -> Run Soft Val -> Map back).
        # Let's do that for maximum speed.
        
        soft_val_prompts = []
        soft_val_indices = [] # (item_idx, candidate_idx)
        
        hard_pass_candidates = [[] for _ in range(len(batch_items))] # List of (text, index) per item
        
        # --- Step 3a: Hard Validation ---
        for j, output in enumerate(gen_outputs):
            item_candidates = candidates_list[j]
            item_latent = batch_items[j]["latent_json"]
            
            for k, cand_output in enumerate(output.outputs):
                text = cand_output.text.strip()
                # Ensure </RESPONSE> is there (vLLM stop might cut it or keep it depending on include_stop_str)
                if not text.endswith("</RESPONSE>"):
                     text += "\n</RESPONSE>"
                     
                is_valid, violations = validator.hard_validate(text, item_latent, item_candidates)
                
                if is_valid:
                    hard_pass_candidates[j].append((text, k))
                # else:
                #     print(f"Hard Fail Item {j} Cand {k}: {violations}")

        # --- Step 3b: Prepare Soft Validation ---
        for j, candidates in enumerate(hard_pass_candidates):
            for text, k in candidates:
                # Prepare Soft Val Prompt
                item = batch_items[j]
                
                # Flatten Latent JSON for clearer prompt to LLM Validator
                flat_latent = flatten_latent(item["latent_json"])
                latent_str = json.dumps(flat_latent, indent=2)
                
                # Get candidates string
                cands_list = candidates_list[j]
                cands_str = ", ".join(cands_list)
                
                val_text = val_template
                val_text = val_text.replace("{question}", item['prompt'])
                val_text = val_text.replace("{candidates}", cands_str)
                val_text = val_text.replace("{latent_json}", latent_str)
                val_text = val_text.replace("{model_output}", text)
                
                # LLM Validator is also Qwen? Yes, re-using same model.
                messages = [
                    {"role": "system", "content": "You are a strict output validator."},
                    {"role": "user", "content": val_text}
                ]
                full_val_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                soft_val_prompts.append(full_val_prompt)
                soft_val_indices.append((j, k, text))
                
        # --- Step 3c: Run Soft Validation (if any) ---
        best_results = [None] * len(batch_items)
        
        if soft_val_prompts:
            # Run validation batch
            val_outputs = llm.generate(soft_val_prompts, val_sampling_params)
            
            # Map back results
            for m, val_out in enumerate(val_outputs):
                j, k, text = soft_val_indices[m]
                val_resp = val_out.outputs[0].text.strip()
                
                # Parse JSON
                try:
                    # Robust JSON cleanup
                    clean_resp = val_resp.strip()
                    # Remove Markdown
                    if "```" in clean_resp:
                        clean_resp = clean_resp.split("```json")[-1].split("```")[0].strip()
                    elif "```" in clean_resp: # Generic block
                        clean_resp = clean_resp.split("```")[-1].split("```")[0].strip()
                    
                    # Fix common trailing comma issue
                    clean_resp = re.sub(r",\s*([}\]])", r"\1", clean_resp)
                    
                    val_json = json.loads(clean_resp)
                    
                    # Enhanced Check Logic
                    checks = val_json.get("checks", {})
                    violations = val_json.get("violations", [])
                    
                    # Pass if ALL checks are True AND no violations listed
                    passed = all(checks.values()) and (len(violations) == 0)
                    
                    if passed:
                        # Success!
                        if best_results[j] is None:
                            best_results[j] = {
                                "text": text,
                                "try_idx": k,
                                "val_checks": checks,
                                "status": "pass",
                                "fail_reason": None
                            }
                    elif best_results[j] is None:
                        # Store soft failure reason if we don't have a success yet
                        best_results[j] = {
                            "text": text,
                            "try_idx": k,
                            "status": "soft_fail",
                            "fail_reason": str(violations) if violations else "Soft checks failed"
                        }
                except Exception as e:
                    # JSON Parse Error Handling
                    # If we can't parse, it's a soft fail.
                    if best_results[j] is None:
                        best_results[j] = {
                            "text": text,
                            "try_idx": k,
                            "status": "soft_fail",
                            "fail_reason": f"Soft Val JSON Parse Error: {e} | Resp: {val_resp[:50]}..."
                        }
        
        # --- Step 4: Finalize & Fallback ---
        for j, item in enumerate(batch_items):
            result = best_results[j]
            
            new_entry = item.copy()
            
            if result and result["status"] == "pass":
                new_entry["generated_reasoning"] = result["text"]
                new_entry["validation_status"] = "pass"
                new_entry["fail_reason"] = None
            elif result and result["status"] == "soft_fail":
                # Soft fail case (Passed Hard, Failed Soft)
                new_entry["generated_reasoning"] = result["text"]
                new_entry["validation_status"] = "fail"
                new_entry["fail_reason"] = result["fail_reason"]
            else:
                # Hard Fail Case (No candidate passed hard val)
                # We need to grab the first candidate and re-run hard val to get the reason
                first_cand = gen_outputs[j].outputs[0].text.strip()
                if not first_cand.endswith("</RESPONSE>"):
                     first_cand += "\n</RESPONSE>"
                     
                item_candidates = candidates_list[j]
                item_latent = item["latent_json"]
                
                _, violations = validator.hard_validate(first_cand, item_latent, item_candidates)
                
                new_entry["generated_reasoning"] = first_cand
                new_entry["validation_status"] = "fail"
                new_entry["fail_reason"] = f"Hard Fail: {str(violations)}"
                
            final_data.append(new_entry)

        # Save Checkpoint
        print(f"Saving checkpoint... Total: {len(final_data)}")
        with open(output_path, "w") as f:
            json.dump(final_data, f, indent=2)
            
    print(f"Part 2 Complete. Final data saved to {output_path}")

def main():
    run_part2("train")

if __name__ == "__main__":
    main()
