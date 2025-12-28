import json
import os
import sys
import re
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from main_qwen import QwenModelManager
from validator import OutputValidator

# --- Constants ---
TEARS_V2_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_V2")

# Reuse Candidates Logic from Part 2
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
    flat = {}
    for cat, content in latent_json.items():
        if isinstance(content, dict):
            for k, v in content.items():
                flat[k] = v
        else:
            flat[cat] = content
    return flat

def run_part3_recovery(split_name: str):
    print(f"=== Part 3: Error Recovery for {split_name} ===")
    
    input_path = TEARS_V2_ROOT / f"{split_name}_v3_final.json"
    output_path = TEARS_V2_ROOT / f"{split_name}_v3_final_recovered.json"
    
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    with open(input_path, "r") as f:
        data = json.load(f)
        
    # Identify failures
    failed_indices = []
    for i, item in enumerate(data):
        if item.get("validation_status") == "fail":
            failed_indices.append(i)
            
    print(f"Total items: {len(data)}")
    print(f"Failed items to recover: {len(failed_indices)}")
    
    if not failed_indices:
        print("No failures found! Copying input to output.")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return

    # --- Initialize vLLM ---
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    llm = QwenModelManager.get_model(model_name)
    tokenizer = llm.get_tokenizer()
    
    # --- Load Templates ---
    # Use the specific prompt for recovery
    template_path = Path(__file__).parent / "data_gen_prompt_after_fail.txt"
    val_template_path = Path(__file__).parent / "data_val_prompt.txt"
    
    if template_path.exists():
        gen_template = template_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError("data_gen_prompt_after_fail.txt not found")
        
    if val_template_path.exists():
        val_template = val_template_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError("data_val_prompt.txt not found")
    
    sys_prompt = "You are a grounded speaker profiling assistant. Fix the previous error."
    validator = OutputValidator()
    
    # --- Sampling Params ---
    gen_sampling_params = SamplingParams(
        temperature=0.8, # Slightly higher temp for recovery to try different paths
        max_tokens=500,
        n=3, # Generate 3 candidates again
        stop=["</RESPONSE>"]
    )
    
    val_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048
    )

    batch_size = 200
    recovered_count = 0
    
    # Process only failed indices in batches
    for i in range(0, len(failed_indices), batch_size):
        batch_idx = failed_indices[i : i + batch_size]
        batch_items = [data[idx] for idx in batch_idx]
        
        print(f"Recovering batch {i//batch_size + 1} ({i}-{i+len(batch_items)})...")
        
        # 1. Format Recovery Prompts
        gen_prompts = []
        candidates_list = []
        
        for item in batch_items:
            latent_str = json.dumps(item["latent_json"], indent=2)
            cands = get_candidates(item['prompt'])
            candidates_list.append(cands)
            cand_str = ", ".join(cands)
            fail_reason = item.get("fail_reason", "Unknown error")
            
            user_text = gen_template
            user_text = user_text.replace("{transcript}", item['transcript'])
            user_text = user_text.replace("{latent_json}", latent_str)
            user_text = user_text.replace("{question}", item['prompt'])
            user_text = user_text.replace("{answer}", item['ground_truth_answer'])
            user_text = user_text.replace("{fail_reason}", str(fail_reason)) # Inject fail reason
            user_text = user_text.replace("{candidate_labels}", cand_str)
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_text},
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            gen_prompts.append(full_prompt)
            
        # 2. Generate
        gen_outputs = llm.generate(gen_prompts, gen_sampling_params)
        
        # 3. Validation
        soft_val_prompts = []
        soft_val_indices = [] 
        hard_pass_candidates = [[] for _ in range(len(batch_items))]
        
        # 3a. Hard Val
        for j, output in enumerate(gen_outputs):
            item_candidates = candidates_list[j]
            item_latent = batch_items[j]["latent_json"]
            
            for k, cand_output in enumerate(output.outputs):
                text = cand_output.text.strip()
                if not text.endswith("</RESPONSE>"):
                     text += "\n</RESPONSE>"
                     
                is_valid, violations = validator.hard_validate(text, item_latent, item_candidates)
                
                if is_valid:
                    hard_pass_candidates[j].append((text, k))

        # 3b. Prepare Soft Val
        for j, candidates in enumerate(hard_pass_candidates):
            for text, k in candidates:
                item = batch_items[j]
                flat_latent = flatten_latent(item["latent_json"])
                latent_str = json.dumps(flat_latent, indent=2)
                
                cands_list = candidates_list[j]
                cands_str = ", ".join(cands_list)
                
                val_text = val_template
                val_text = val_text.replace("{question}", item['prompt'])
                val_text = val_text.replace("{candidates}", cands_str)
                val_text = val_text.replace("{latent_json}", latent_str)
                val_text = val_text.replace("{model_output}", text)
                
                messages = [
                    {"role": "system", "content": "You are a strict output validator."},
                    {"role": "user", "content": val_text}
                ]
                full_val_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                soft_val_prompts.append(full_val_prompt)
                soft_val_indices.append((j, k, text))
                
        # 3c. Run Soft Val
        best_results = [None] * len(batch_items)
        
        if soft_val_prompts:
            val_outputs = llm.generate(soft_val_prompts, val_sampling_params)
            
            for m, val_out in enumerate(val_outputs):
                j, k, text = soft_val_indices[m]
                val_resp = val_out.outputs[0].text.strip()
                
                try:
                    clean_resp = val_resp.strip()
                    if "```" in clean_resp:
                        clean_resp = clean_resp.split("```json")[-1].split("```")[0].strip()
                    elif "```" in clean_resp:
                        clean_resp = clean_resp.split("```")[-1].split("```")[0].strip()
                    clean_resp = re.sub(r",\s*([}\]])", r"\1", clean_resp)
                    
                    val_json = json.loads(clean_resp)
                    checks = val_json.get("checks", {})
                    violations = val_json.get("violations", [])
                    
                    passed = all(checks.values()) and (len(violations) == 0)
                    
                    if passed:
                        if best_results[j] is None:
                            best_results[j] = {
                                "text": text,
                                "status": "pass",
                                "fail_reason": None
                            }
                    elif best_results[j] is None:
                        best_results[j] = {
                            "text": text,
                            "status": "soft_fail",
                            "fail_reason": str(violations) if violations else "Soft checks failed"
                        }
                except Exception as e:
                    if best_results[j] is None:
                        best_results[j] = {
                            "text": text,
                            "status": "soft_fail",
                            "fail_reason": f"JSON Error: {e}"
                        }

        # 4. Update Data (In-Place in 'data' list)
        for j, item in enumerate(batch_items):
            original_idx = batch_idx[j]
            result = best_results[j]
            
            if result and result["status"] == "pass":
                data[original_idx]["generated_reasoning"] = result["text"]
                data[original_idx]["validation_status"] = "pass"
                data[original_idx]["fail_reason"] = None
                recovered_count += 1
            else:
                # Still failed - keep new attempt if it was a soft fail (might be better), or keep old?
                # Let's keep the NEW attempt to see why it failed again.
                if result:
                    data[original_idx]["generated_reasoning"] = result["text"]
                    data[original_idx]["validation_status"] = "fail"
                    data[original_idx]["fail_reason"] = result["fail_reason"]
                else:
                    # Hard fail again
                    if gen_outputs[j].outputs:
                        data[original_idx]["generated_reasoning"] = gen_outputs[j].outputs[0].text
                        data[original_idx]["fail_reason"] = "Recovery Hard Fail"
                    
        # Checkpoint
        print(f"Saving recovery checkpoint... Recovered so far: {recovered_count}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
    print(f"Part 3 Complete. Total recovered: {recovered_count} / {len(failed_indices)}")
    print(f"Final data saved to {output_path}")

def main():
    run_part3_recovery("train")

if __name__ == "__main__":
    main()


