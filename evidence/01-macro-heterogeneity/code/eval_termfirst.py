#!/usr/bin/env python3
"""
CLEAN term-first evaluation on AIME2024 with Qwen2.5-3B-Instruct via vLLM.
n=64 pass@1. 2 runs per condition (Standard + Term-first).
NO cherry-picking. ALL runs reported. Pre-registered design.

Protocol:
  - Standard: No system prompt. Question + "Please put your final answer within \\boxed{}."
  - Term-first: System prompt "What is the query? Term first, then reasoning." + question + boxed instruction.
  - Fixed: T=0.6, top_p=0.95, max_tokens=16384
  - Seeds: 42,43 (Standard); 44,45 (Term-first)
"""

import json
import os
import re
import sys
import time
import csv
import asyncio
import aiohttp
from datetime import datetime

# ========== FIXED PARAMETERS ==========
VLLM_URL = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct/"
DATASET_PATH = "/root/autodl-tmp/aime2024.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/termfirst_clean_eval_v2/"
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 16384
N_PER_QUESTION = 64  # pass@1 per question
N_QUESTIONS = 30

# 2 runs per condition
RUN_CONFIGS = [
    {"condition": "Standard", "run": "Run1", "seed": 42, "system_prompt": None},
    {"condition": "Standard", "run": "Run2", "seed": 43, "system_prompt": None},
    {"condition": "Term-first", "run": "Run1", "seed": 44, "system_prompt": "What is the query? Term first, then reasoning."},
    {"condition": "Term-first", "run": "Run2", "seed": 45, "system_prompt": "What is the query? Term first, then reasoning."},
]

# Max concurrent requests to vLLM
MAX_CONCURRENT = 32  # reduced from 64 for stability

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========== ANSWER EXTRACTION ==========
def extract_boxed(text):
    """Extract the final answer from \\boxed{...} with nested brace handling."""
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
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
    """Normalize numeric answer for comparison."""
    if ans is None:
        return None
    ans = ans.strip()
    ans = ans.replace(',', '')
    ans = ans.replace('\\text{', '').replace('}', '')
    ans = ans.replace('$', '')
    # Try to extract integer
    match = re.search(r'-?\d+', ans)
    if match:
        return match.group(0)
    return ans.strip().lower()


# ========== DATA LOADING ==========
with open(DATASET_PATH) as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} questions from AIME2024")
for d in dataset:
    print(f"  {d['ID']}: Answer={d['Answer']}")


# ========== ASYNC API CALL ==========
async def generate_one(session, semaphore, question, system_prompt, seed, idx):
    """Generate one response from vLLM API."""
    async with semaphore:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        prompt = f"{question}\n\nPlease put your final answer within \\boxed{{}}."
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": VLLM_MODEL,
            "messages": messages,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "seed": seed,
        }

        for attempt in range(3):
            try:
                async with session.post(
                    VLLM_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600, connect=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        text = await resp.text()
                        print(f"    API error (attempt {attempt+1}): {resp.status} - {text[:200]}", flush=True)
                        await asyncio.sleep(2 ** attempt)
            except asyncio.TimeoutError:
                print(f"    Timeout (attempt {attempt+1})", flush=True)
                await asyncio.sleep(2 ** attempt)
            except aiohttp.ClientError as e:
                print(f"    Client error (attempt {attempt+1}): {type(e).__name__}: {e}", flush=True)
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"    Request error (attempt {attempt+1}): {type(e).__name__}: {e}", flush=True)
                await asyncio.sleep(2 ** attempt)
        return f"GENERATION_ERROR after 3 attempts"


# ========== EVALUATE ONE RUN ==========
async def evaluate_run(config, dataset, run_idx, total_runs):
    """Evaluate a single run: N_PER_QUESTION generations per question."""
    cond = config["condition"]
    run_label = config["run"]
    seed = config["seed"]
    sys_prompt = config["system_prompt"]

    print(f"\n{'='*60}")
    print(f"RUN {run_idx+1}/{total_runs}: {cond} {run_label} (seed={seed})")
    print(f"System prompt: {repr(sys_prompt)}")
    print(f"Total: {N_PER_QUESTION} generations x {N_QUESTIONS} questions = {N_PER_QUESTION * N_QUESTIONS} gens")
    print(f"{'='*60}")

    # Checkpoint file for this run
    checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_{cond}_{run_label}.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            run_results = json.load(f)
        completed_questions = len(run_results)
        print(f"Resuming from checkpoint: {completed_questions} questions already done")
    else:
        run_results = []
        completed_questions = 0

    run_start = time.time()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        for q_idx in range(completed_questions, N_QUESTIONS):
            item = dataset[q_idx]
            question = item["Problem"]
            ground_truth = str(item["Answer"])
            qid = item["ID"]

            print(f"\n  [{run_idx+1}/{total_runs}] Q{q_idx+1}/{N_QUESTIONS} ({qid}) gt={ground_truth}", flush=True)

            # Fire concurrent requests with error handling
            q_start = time.time()
            tasks = []
            for i in range(N_PER_QUESTION):
                gen_seed = seed * 10000 + i  # unique seed per generation
                tasks.append(generate_one(session, semaphore, question, sys_prompt, gen_seed, i))

            try:
                raw_outputs = await asyncio.gather(*tasks)
            except Exception as e:
                print(f"  FATAL: asyncio.gather failed: {e}", flush=True)
                raw_outputs = [f"GATHER_ERROR: {e}"] * N_PER_QUESTION
            q_elapsed = time.time() - q_start

            # Score
            correct_count = 0
            gen_details = []
            for i, raw in enumerate(raw_outputs):
                boxed = extract_boxed(raw)
                norm_boxed = normalize_answer(boxed)
                is_correct = (norm_boxed == ground_truth)
                if is_correct:
                    correct_count += 1
                gen_details.append({
                    "gen_idx": i,
                    "seed": seed * 10000 + i,
                    "raw_output": raw,
                    "extracted_boxed": boxed,
                    "normalized_boxed": norm_boxed,
                    "correct": is_correct,
                })

            acc = correct_count / N_PER_QUESTION
            print(f"  => {correct_count}/{N_PER_QUESTION} correct ({acc*100:.1f}%) in {q_elapsed:.0f}s ({q_elapsed/N_PER_QUESTION:.1f}s/gen)", flush=True)

            run_results.append({
                "question_idx": q_idx,
                "question_id": qid,
                "ground_truth": ground_truth,
                "n_generations": N_PER_QUESTION,
                "correct_count": correct_count,
                "accuracy": round(acc, 6),
                "elapsed_sec": round(q_elapsed, 1),
                "generations": gen_details,
            })

            # Checkpoint after each question
            with open(checkpoint_path + ".tmp", "w") as f:
                json.dump(run_results, f, indent=2, ensure_ascii=False, default=str)
            os.rename(checkpoint_path + ".tmp", checkpoint_path)
            print(f"    Checkpoint saved")

    run_elapsed = time.time() - run_start

    # Compute overall metrics
    per_q_accs = [r["accuracy"] for r in run_results]
    mean_acc = sum(per_q_accs) / len(per_q_accs)
    total_correct = sum(r["correct_count"] for r in run_results)
    total_gens = N_PER_QUESTION * N_QUESTIONS

    print(f"\n  {cond} {run_label} FINAL:")
    print(f"    Per-question mean pass@1: {mean_acc*100:.2f}%")
    print(f"    Total correct: {total_correct}/{total_gens} ({total_correct/total_gens*100:.1f}%)")
    print(f"    Wall time: {run_elapsed:.0f}s ({run_elapsed/60:.1f}min)")

    return {
        "condition": cond,
        "run": run_label,
        "seed": seed,
        "per_question_accuracies": per_q_accs,
        "mean_accuracy": round(mean_acc, 6),
        "total_correct": total_correct,
        "total_generations": total_gens,
        "elapsed_sec": round(run_elapsed, 1),
        "results": run_results,
    }


# ========== MAIN ==========
async def main():
    print(f"Starting term-first clean eval at {datetime.now().isoformat()}")
    print(f"vLLM API: {VLLM_URL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Configs: {len(RUN_CONFIGS)} runs total")
    print()

    # Test API connectivity
    print("Testing vLLM API connectivity...")
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            test_payload = {
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 16,
                "temperature": 0.0,
            }
            async with session.post(VLLM_URL, json=test_payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response_text = data["choices"][0]["message"]["content"]
                    print(f"  API OK: '{response_text.strip()}'")
                else:
                    text = await resp.text()
                    print(f"  API FAILED: {resp.status} - {text[:300]}")
                    return
        except Exception as e:
            print(f"  API CONNECTION FAILED: {e}")
            return

    all_summaries = []

    for run_idx, config in enumerate(RUN_CONFIGS):
        summary = await evaluate_run(config, dataset, run_idx, len(RUN_CONFIGS))
        all_summaries.append(summary)

        # Save partial summary after each run
        partial_summary_path = os.path.join(OUTPUT_DIR, f"summary_partial_{run_idx+1}.csv")
        with open(partial_summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["condition", "run", "seed", "mean_accuracy", "total_correct", "total_generations", "elapsed_sec"])
            writer.writeheader()
            for s in all_summaries:
                writer.writerow({k: s[k] for k in writer.fieldnames})
        print(f"  Partial summary saved to {partial_summary_path}")

    # ========== FINAL SUMMARY ==========
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — ALL RUNS REPORTED")
    print(f"{'='*60}")

    for cond_name in ["Standard", "Term-first"]:
        accs = [s["mean_accuracy"] for s in all_summaries if s["condition"] == cond_name]
        mean_acc = sum(accs) / len(accs)
        # std across runs (not per-question std across runs)
        var = sum((a - mean_acc)**2 for a in accs) / len(accs)
        std_acc = var ** 0.5
        print(f"\n{cond_name}:")
        for i, acc in enumerate(accs):
            print(f"  Run{i+1}: {acc*100:.2f}%")
        print(f"  Mean: {mean_acc*100:.2f}% ± {std_acc*100:.2f}% (std across runs)")

    std_accs = [s["mean_accuracy"] for s in all_summaries if s["condition"] == "Standard"]
    tf_accs = [s["mean_accuracy"] for s in all_summaries if s["condition"] == "Term-first"]
    std_mean = sum(std_accs) / len(std_accs)
    tf_mean = sum(tf_accs) / len(tf_accs)
    diff = (tf_mean - std_mean) * 100
    print(f"\nDifference (Term-first - Standard): {diff:+.2f} percentage points")

    # ========== SAVE OUTPUTS ==========
    # 1. Full detailed results (with all raw outputs — could be large)
    detailed_path = os.path.join(OUTPUT_DIR, "detailed_results.json")
    print(f"\nSaving detailed results to {detailed_path}...")
    with open(detailed_path, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False, default=str)
    print("  Done.")

    # 2. Compact results (without raw outputs, easier to work with)
    compact = []
    for s in all_summaries:
        compact_results = []
        for r in s["results"]:
            compact_results.append({
                "question_idx": r["question_idx"],
                "question_id": r["question_id"],
                "ground_truth": r["ground_truth"],
                "correct_count": r["correct_count"],
                "accuracy": r["accuracy"],
                "n_generations": r["n_generations"],
            })
        compact.append({
            "condition": s["condition"],
            "run": s["run"],
            "seed": s["seed"],
            "mean_accuracy": s["mean_accuracy"],
            "total_correct": s["total_correct"],
            "total_generations": s["total_generations"],
            "elapsed_sec": s["elapsed_sec"],
            "per_question": compact_results,
        })
    compact_path = os.path.join(OUTPUT_DIR, "compact_results.json")
    with open(compact_path, "w") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)
    print(f"Compact results saved to {compact_path}")

    # 3. Summary CSV
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "run", "seed", "mean_accuracy", "total_correct", "total_generations", "elapsed_sec"])
        writer.writeheader()
        for s in all_summaries:
            writer.writerow({k: s[k] for k in writer.fieldnames})
    print(f"Summary saved to {summary_path}")

    # 4. Per-question breakdown CSV
    pq_path = os.path.join(OUTPUT_DIR, "per_question.csv")
    with open(pq_path, "w", newline="") as f:
        fieldnames = ["condition", "run", "seed", "question_idx", "question_id", "ground_truth", "correct_count", "n_generations", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_summaries:
            cond = s["condition"]
            run = s["run"]
            seed = s["seed"]
            for r in s["results"]:
                writer.writerow({
                    "condition": cond,
                    "run": run,
                    "seed": seed,
                    "question_idx": r["question_idx"],
                    "question_id": r["question_id"],
                    "ground_truth": r["ground_truth"],
                    "correct_count": r["correct_count"],
                    "n_generations": r["n_generations"],
                    "accuracy": r["accuracy"],
                })
    print(f"Per-question breakdown saved to {pq_path}")

    # 5. README.md
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        now = datetime.now().isoformat()
        f.write(f"""# Term-First Clean Evaluation on AIME2024

## Experiment Log

- **Date completed:** {now}
- **Model:** Qwen2.5-3B-Instruct
- **Dataset:** AIME2024 (30 questions)
- **Protocol:** n=64 pass@1 (64 independent generations per question)
- **Runs:** 2 Standard + 2 Term-first = 4 total
- **Total generations:** {N_PER_QUESTION * N_QUESTIONS * len(RUN_CONFIGS)}
- **Temperature:** {TEMPERATURE}
- **Top-p:** {TOP_P}
- **Max tokens:** {MAX_TOKENS}
- **Seeds:** Standard: 42, 43; Term-first: 44, 45
- **vLLM URL:** {VLLM_URL}

## Conditions

### Standard
No system prompt. Question text + "Please put your final answer within \\boxed{{}}."

### Term-first
System prompt: "What is the query? Term first, then reasoning." + question text + boxed instruction.

## Results

| Condition | Run | Seed | Accuracy | Correct/Total | Elapsed (s) |
|-----------|-----|------|----------|---------------|-------------|
""")
        for s in all_summaries:
            f.write(f"| {s['condition']} | {s['run']} | {s['seed']} | {s['mean_accuracy']*100:.2f}% | {s['total_correct']}/{s['total_generations']} | {s['elapsed_sec']:.0f} |\n")

        f.write(f"""
## Summary

""")
        for cond_name in ["Standard", "Term-first"]:
            accs = [s["mean_accuracy"] for s in all_summaries if s["condition"] == cond_name]
            mean_acc = sum(accs) / len(accs)
            var = sum((a - mean_acc)**2 for a in accs) / len(accs)
            std_acc = var ** 0.5
            f.write(f"- **{cond_name}:** Mean = {mean_acc*100:.2f}% ± {std_acc*100:.2f}% (std across {len(accs)} runs)\n")
            for i, acc in enumerate(accs):
                f.write(f"  - Run{i+1}: {acc*100:.2f}%\n")

        f.write(f"""
- **Difference (Term-first - Standard):** {diff:+.2f} pp

## Notes

- ALL 4 runs reported — no cherry-picking
- pass@1 metric: per-question fraction of 64 generations with correct answer
- Overall accuracy: mean of 30 per-question pass@1s (not micro-average)
- Answer extraction: last `\\boxed{{...}}` in response
- Comparison: exact match after normalizing answer and ground truth to integers
""")

    print(f"README saved to {readme_path}")
    print("\n" + "="*60)
    print("ALL DONE.")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
