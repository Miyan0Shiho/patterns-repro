import json
import numpy as np
import os

def verify_data():
    _HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(_HERE, "..", "data", "real_logit_lens_data.json")
    data_path = os.path.normpath(data_path)
    if not os.path.exists(data_path):
        print("Data file not found.")
        return
        
    with open(data_path, "r") as f:
        data = json.load(f)
        
    print("=== Real Logit Lens Data Verification ===")
    
    for condition, items in data.items():
        print(f"\nCondition: {condition.upper()}")
        print(f"Number of samples: {len(items)}")
        
        all_probs = []
        for item in items:
            probs = item["layer_probs"]
            all_probs.append(probs)
            print(f"  - Prompt {item['prompt_idx']}: target='{item['label']}' (token '{item['token_str']}'), layers={len(probs)}, max_prob={max(probs):.4f}")
            
        # Calculate mean across samples if they have the same number of layers
        # Qwen 1.5B has 28 layers, plus embedding = 29 layers
        if all_probs:
            min_len = min(len(p) for p in all_probs)
            truncated_probs = [p[:min_len] for p in all_probs]
            mean_probs = np.mean(truncated_probs, axis=0)
            print(f"  => Mean trajectory: start={mean_probs[0]:.4f} -> mid={mean_probs[min_len//2]:.4f} -> end={mean_probs[-1]:.4f}")

if __name__ == "__main__":
    verify_data()
