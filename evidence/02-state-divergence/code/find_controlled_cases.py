import os
import json
from collections import defaultdict
import re

# ARCHIVAL: Requires external AIME prediction data not in this repository.
# The selected controlled case is saved in ../data/selected_controlled_case.json.
# data_dir = "/path/to/aime2024_picture/data/deepseek-r1-1.5b-aime2024"
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                         "..", "Desktop", "aime2024_picture", "data", "deepseek-r1-1.5b-aime2024")

question_stats = defaultdict(lambda: defaultdict(list))
subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for subdir in subdirs:
    pred_dir = os.path.join(data_dir, subdir, "predictions")
    if not os.path.exists(pred_dir): continue
    
    for model_dir in os.listdir(pred_dir):
        if model_dir == ".DS_Store": continue
        match = re.search(r'_1_5B_(\d+)_models', model_dir)
        prompt_idx = int(match.group(1)) if match else 0
        
        full_model_dir = os.path.join(pred_dir, model_dir)
        for json_file in os.listdir(full_model_dir):
            if not json_file.endswith(".json"): continue
            
            with open(os.path.join(full_model_dir, json_file), "r") as f:
                data = json.load(f)
                
            # data is usually a dict for OpenCompass predictions
            iterator = data.values() if isinstance(data, dict) else data
            
            for item in iterator:
                q_text = item.get('origin_prompt', str(item))
                if isinstance(q_text, list): 
                    # extract content from chat template
                    try:
                        q_text = q_text[0]['content']
                    except:
                        q_text = str(q_text)
                    
                pred = str(item.get('prediction', ''))
                
                # We need gold to know correctness
                gold = item.get('gold', '')
                
                length = len(pred)
                wait_count = pred.lower().count("wait")
                
                question_stats[q_text][prompt_idx].append({
                    "pred": pred,
                    "len": length,
                    "wait": wait_count,
                    "gold": gold,
                    "item": item
                })

# Find a question that appears in multiple prompts
good_questions = []
for q_text, prompts in question_stats.items():
    if len(prompts) > 5: # question was evaluated with many prompts
        good_questions.append((q_text, prompts))

print(f"Found {len(good_questions)} questions evaluated across multiple prompts.")

if good_questions:
    q_text, prompts = good_questions[0]
    print(f"\nTarget Question: {q_text[:100]}...\n")
    
    for p_idx, results in prompts.items():
        avg_len = sum(r['len'] for r in results) / len(results)
        avg_wait = sum(r['wait'] for r in results) / len(results)
        print(f"Prompt {p_idx}: {len(results)} samples, avg_len={avg_len:.1f}, avg_wait={avg_wait:.1f}")
        
    # Pick 3 specific responses for this question:
    # 1. Under-thinking (shortest length)
    # 2. Normal-thinking (medium length)
    # 3. Over-thinking (longest length / most waits)
    
    all_responses = []
    for p_idx, results in prompts.items():
        for r in results:
            all_responses.append((p_idx, r))
            
    all_responses.sort(key=lambda x: x[1]['len'])
    
    under = all_responses[0]
    over = all_responses[-1]
    normal = all_responses[len(all_responses)//2]
    
    print("\n--- Selected Cases ---")
    print(f"Under-thinking: Prompt {under[0]}, len={under[1]['len']}, wait={under[1]['wait']}")
    print(f"Normal-thinking: Prompt {normal[0]}, len={normal[1]['len']}, wait={normal[1]['wait']}")
    print(f"Over-thinking: Prompt {over[0]}, len={over[1]['len']}, wait={over[1]['wait']}")
    
    # Save the selected question and prompt indices
    output = {
        "question": q_text,
        "cases": {
            "under_thinking": {"prompt_idx": under[0], "len": under[1]['len'], "wait": under[1]['wait']},
            "normal_thinking": {"prompt_idx": normal[0], "len": normal[1]['len'], "wait": normal[1]['wait']},
            "over_thinking": {"prompt_idx": over[0], "len": over[1]['len'], "wait": over[1]['wait']}
        }
    }
    with open("workspace/selected_controlled_case.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to workspace/selected_controlled_case.json")
