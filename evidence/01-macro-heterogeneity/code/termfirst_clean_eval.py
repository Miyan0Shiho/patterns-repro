#!/usr/bin/env python3
"""
CLEAN term-first evaluation on AIME2024 with Qwen2.5-3B-Instruct.
NO cherry-picking. ALL runs reported. PRE-REGISTERED design.

Conditions:
  - Standard: No system prompt, just the question.
  - Term-first: System prompt "What is the query? Term first, then reasoning."

4 independent runs per condition (seeds 42, 43, 44, 45).
30 AIME2024 questions each. pass@1 via \boxed{} extraction.
"""

import json
import os
import re
import sys
import time
import csv
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== FIXED PARAMETERS (DO NOT CHANGE) ==========
MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct/"
DATASET_PATH = "/root/autodl-tmp/aime2024.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/termfirst_clean_eval/"
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 16384
NUM_RUNS = 4
SEEDS = [42, 43, 44, 45]

CONDITIONS = {
    "Standard": None,  # No system prompt
    "Term-first": "What is the query? Term first, then reasoning.",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== DATA LOADING ==========
with open(DATASET_PATH) as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} questions from AIME2024")
for i, d in enumerate(dataset):
    print(f"  [{i}] {d['ID']} Answer={d['Answer']}")

# ========== MODEL LOADING ==========
print(f"\nLoading model from {MODEL_PATH}...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

print(f"Model loaded in {time.time() - t0:.1f}s")
print(f"Model device: {model.device}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ========== ANSWER EXTRACTION ==========
def extract_boxed(text):
    """
    Extract the final answer from \\boxed{...}.
    Returns the contents of the LAST \\boxed{} in the text.
    """
    # Pattern: \boxed{...} handling nested braces
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Use the last match
    last_match = matches[-1]
    start = last_match.end()
    brace_count = 1
    end = start
    while end < len(text) and brace_count > 0:
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
            if brace_count == 0:
                break
        end += 1

    if brace_count == 0:
        return text[start:end].strip()
    return None


def normalize_answer(ans):
    """Normalize a numeric answer for comparison."""
    if ans is None:
        return None
    ans = ans.strip()
    # Remove commas (e.g., "1,234" -> "1234")
    ans = ans.replace(',', '')
    # Remove LaTeX formatting
    ans = ans.replace('\\text{', '').replace('}', '')
    ans = ans.replace('$', '')
    # Try to extract just the number
    # Match integer
    match = re.search(r'-?\d+', ans)
    if match:
        return match.group(0)
    return ans.strip()


# ========== GENERATION ==========
def generate_one(model, tokenizer, question, system_prompt=None, seed=None, answer_key=None):
    """
    Generate an answer for a single question.
    Returns the raw output text.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build prompt: question + boxed instruction
    prompt = f"{question}\n\nPlease put your final answer within \\boxed{{}}."
    messages.append({"role": "user", "content": prompt})

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


# ========== MAIN EVALUATION LOOP ==========
all_results = {}  # condition -> run -> list of per-question results
summary_rows = []

total_tasks = len(CONDITIONS) * NUM_RUNS * len(dataset)
task_idx = 0

for cond_name, sys_prompt in CONDITIONS.items():
    print(f"\n{'='*60}")
    print(f"CONDITION: {cond_name}")
    print(f"System prompt: {repr(sys_prompt)}")
    print(f"{'='*60}")

    all_results[cond_name] = {}

    for run_num in range(1, NUM_RUNS + 1):
        seed = SEEDS[run_num - 1]
        run_label = f"Run{run_num}"
        print(f"\n--- {cond_name} {run_label} (seed={seed}) ---")

        run_results = []
        num_correct = 0
        run_start = time.time()

        for q_idx, item in enumerate(dataset):
            task_idx += 1
            question = item["Problem"]
            ground_truth = str(item["Answer"])
            qid = item["ID"]

            print(f"  [{task_idx}/{total_tasks}] Q{q_idx+1}/30 ({qid})...", end=" ", flush=True)

            q_start = time.time()
            try:
                raw_output = generate_one(
                    model, tokenizer, question,
                    system_prompt=sys_prompt,
                    seed=seed,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                raw_output = f"GENERATION_ERROR: {e}"

            q_elapsed = time.time() - q_start
            boxed = extract_boxed(raw_output)
            norm_boxed = normalize_answer(boxed)
            is_correct = (norm_boxed == ground_truth)

            if is_correct:
                num_correct += 1
                print(f"CORRECT (boxed={boxed}, gt={ground_truth}, {q_elapsed:.0f}s)")
            else:
                print(f"WRONG (boxed={boxed}, gt={ground_truth}, {q_elapsed:.0f}s)")

            run_results.append({
                "question_idx": q_idx,
                "question_id": qid,
                "ground_truth": ground_truth,
                "raw_output": raw_output,
                "extracted_boxed": boxed,
                "normalized_boxed": norm_boxed,
                "correct": is_correct,
                "elapsed_sec": round(q_elapsed, 1),
            })

        run_elapsed = time.time() - run_start
        accuracy = num_correct / len(dataset)
        print(f"\n  {cond_name} {run_label}: {num_correct}/{len(dataset)} correct = {accuracy:.4f} ({accuracy*100:.1f}%) in {run_elapsed:.0f}s")

        all_results[cond_name][run_label] = run_results
        summary_rows.append({
            "condition": cond_name,
            "run": run_label,
            "seed": seed,
            "accuracy": round(accuracy, 6),
            "num_correct": num_correct,
            "num_total": len(dataset),
            "elapsed_sec": round(run_elapsed, 1),
        })

# ========== FINAL SUMMARY ==========
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")

for cond_name in ["Standard", "Term-first"]:
    accs = [summary_rows[i]["accuracy"] for i in range(len(summary_rows))
            if summary_rows[i]["condition"] == cond_name]
    mean_acc = sum(accs) / len(accs)
    std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
    print(f"\n{cond_name}:")
    for i, acc in enumerate(accs):
        print(f"  Run{i+1}: {acc*100:.1f}%")
    print(f"  Mean: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")

# Difference
std_accs = [summary_rows[i]["accuracy"] for i in range(len(summary_rows))
            if summary_rows[i]["condition"] == "Standard"]
tf_accs = [summary_rows[i]["accuracy"] for i in range(len(summary_rows))
           if summary_rows[i]["condition"] == "Term-first"]
std_mean = sum(std_accs) / len(std_accs)
tf_mean = sum(tf_accs) / len(tf_accs)
diff = (tf_mean - std_mean) * 100
print(f"\nDifference (Term-first - Standard): {diff:+.1f} percentage points")

# ========== SAVE OUTPUTS ==========
# 1. Detailed results JSON
detailed_path = os.path.join(OUTPUT_DIR, "detailed_results.json")
with open(detailed_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nDetailed results saved to {detailed_path}")

# 2. Summary CSV
summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["condition", "run", "seed", "accuracy", "num_correct", "num_total", "elapsed_sec"])
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"Summary saved to {summary_path}")

# 3. README.md
readme_path = os.path.join(OUTPUT_DIR, "README.md")
with open(readme_path, "w") as f:
    f.write(f"""# Term-First Clean Evaluation on AIME2024

## Experiment Log

- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model:** Qwen2.5-3B-Instruct from `{MODEL_PATH}`
- **Dataset:** AIME2024 (30 questions)
- **Temperature:** {TEMPERATURE}
- **Top-p:** {TOP_P}
- **Max new tokens:** {MAX_NEW_TOKENS}
- **Seeds:** {SEEDS}
- **Python:** {sys.version}
- **Torch:** {torch.__version__}
- **GPU:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}

## Conditions

### Condition A: Standard
No system prompt. Just the question text + "Please put your final answer within \\boxed{{}}."

### Condition B: Term-first
System prompt: "What is the query? Term first, then reasoning." Then the question text + boxed instruction.

## Results

| Condition | Run | Seed | Accuracy | Correct | Total | Elapsed (s) |
|-----------|-----|------|----------|---------|-------|-------------|
""")
    for row in summary_rows:
        f.write(f"| {row['condition']} | {row['run']} | {row['seed']} | {row['accuracy']*100:.1f}% | {row['num_correct']}/{row['num_total']} | {row['elapsed_sec']:.0f} |\n")

    f.write(f"""
## Summary

- **Standard:** Mean = {std_mean*100:.1f}% ± {sum((a - std_mean)**2 for a in std_accs)**0.5 / len(std_accs)**0.5 * 100 if len(std_accs) > 1 else 0:.1f}%
- **Term-first:** Mean = {tf_mean*100:.1f}% ± {sum((a - tf_mean)**2 for a in tf_accs)**0.5 / len(tf_accs)**0.5 * 100 if len(tf_accs) > 1 else 0:.1f}%
- **Difference (Term-first - Standard):** {diff:+.1f} pp

## Notes

- pass@1 metric: exact match after extracting from \\\\boxed{{}}
- All 8 runs (4 Standard + 4 Term-first) on all 30 questions are reported
- No cherry-picking — all runs included
""")

print(f"README saved to {readme_path}")
print("\nDONE.")
