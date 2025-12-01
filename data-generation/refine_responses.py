import json
from pathlib import Path
from tqdm import tqdm

# Configuration
TEARS_V2_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_V2")

def clean_response(response: str, answer: str) -> str:
    """
    Clean the response string to remove artifacts like:
    - (Hidden gold; do not reference or copy.)
    - </FINAL_ANSWER> tags
    - Trailing whitespace/newlines
    - Ensure it ends with the answer if possible, or just clean up.
    """
    if not response:
        return ""
        
    cleaned = response
    
    # 1. Remove specific garbage strings
    garbage_strings = [
        "(Hidden gold; do not reference or copy.)",
        "</FINAL_ANSWER>",
        "<|endoftext|>",
    ]
    
    for g in garbage_strings:
        cleaned = cleaned.replace(g, "")
        
    # 2. Strip whitespace
    cleaned = cleaned.strip()
    
    # 3. Ensure it ends cleanly. 
    # The user said: "response 끝에가 answer로 끝날 수 있게 그 뒷부분은 날려주는 코드"
    # This means if the ground truth 'answer' appears in the response, we should cut off anything AFTER it.
    # But we need to be careful not to cut off if the answer appears earlier in the text by coincidence.
    # Typically, <FINAL_ANSWER> prompts end with the answer.
    # Let's find the LAST occurrence of the answer string (case insensitive?)
    # Or just strict string matching at the end.
    
    # If the response *already* ends with the answer (ignoring punctuation/case), great.
    # If not, look for the answer near the end and truncate after it.
    
    if not answer:
        return cleaned
        
    # Normalization for matching
    # We want to find `answer` in `cleaned` and cut everything after it.
    # Case insensitive search for robustness
    
    idx = cleaned.lower().rfind(answer.lower())
    
    if idx != -1:
        # Found the answer.
        # Cut after the answer
        cutoff_index = idx + len(answer)
        cleaned = cleaned[:cutoff_index]
        
    return cleaned.strip()

def process_file(filename: str):
    input_path = TEARS_V2_ROOT / f"{filename}_v2.json"
    output_path = TEARS_V2_ROOT / f"{filename}_v3.json"
    
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Processing {filename}...")
    try:
        with open(input_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return

    cleaned_data = []
    modified_count = 0
    
    for item in tqdm(data):
        original_response = item.get("response", "")
        answer = item.get("answer", "")
        
        cleaned_resp = clean_response(original_response, answer)
        
        if cleaned_resp != original_response:
            modified_count += 1
            
        item["response"] = cleaned_resp
        cleaned_data.append(item)
        
    print(f"Saving {len(cleaned_data)} items to {output_path} (Modified: {modified_count})")
    
    with open(output_path, "w") as f:
        json.dump(cleaned_data, f, indent=2)

def main():
    process_file("train")
    process_file("test")
    print("Done.")

if __name__ == "__main__":
    main()

