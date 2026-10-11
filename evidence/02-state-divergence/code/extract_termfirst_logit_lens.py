"""Logit Lens extraction: 4-condition controlled experiment.

Adds term-first as the 4th condition to the existing 3-condition setup.
Same question, same model, same extraction method — only the prompt prefix changes.

Run on server with CUDA. MPS fallback available for local testing.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

TARGET_QUESTION = (
    "Let $x,y$ and $z$ be positive real numbers that satisfy the following "
    "system of equations: \n"
    "\\[\\log_2\\left({x \\over yz}\\right) = {1 \\over 2}\\]\n"
    "\\[\\log_2\\left({y \\over xz}\\right) = {1 \\over 3}\\]\n"
    "\\[\\log_2\\left({z \\over xy}\\right) = {1 \\over 4}\\]\n"
    "Then the value of $\\left|\\log_2(x^4y^3z^2)\\right|$ is $\\tfrac{m}{n}$ "
    "where $m$ and $n$ are relatively prime positive integers. Find $m+n$.\n"
    "Please put your final answer within \\boxed{}."
)

TERM_FIRST_PROMPT = "What is the query? Term first, then reasoning."

# All 4 conditions: under / normal / over / term-first
PROMPTS = {
    "under_thinking": ["Wait, think.", TARGET_QUESTION],
    "normal_thinking": ["Let's reason through this step by step.", TARGET_QUESTION],
    "over_thinking": ["Let's think step by step.", TARGET_QUESTION],
    "term_first": [TERM_FIRST_PROMPT, TARGET_QUESTION],
}


def get_token_id(tokenizer, text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens[0] if tokens else None


def run_extraction(device: str = "cuda", max_new_tokens: int = 32000):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model on {device} (float16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device if device != "mps" else "mps",
    )
    if device == "mps":
        model = model.to("mps")
    model.eval()

    results = {}

    for condition, (cot_prompt, question) in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {condition}")
        print(f"Prompt prefix: {cot_prompt[:80]}...")

        full_prompt = f"{cot_prompt}\n\n{question}"
        messages = [{"role": "user", "content": full_prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        print(f"Generating (max_new_tokens={max_new_tokens})...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        prompt_len = inputs.input_ids.shape[1]
        gen_only_ids = generated_ids[prompt_len:]

        print(f"Total output length: {len(generated_text)} chars")
        print(f"Generated tokens: {len(gen_only_ids)}")

        # ---- Target token selection ----
        target_idx = None
        target_token_id = None
        target_label = ""

        think_end = tokenizer.encode("</think>", add_special_tokens=False)

        if condition == "over_thinking":
            # Find last 'Wait' token (backward search)
            wait_tok = get_token_id(tokenizer, "Wait")
            wait_tok_2 = get_token_id(tokenizer, " Wait")
            for i in range(len(gen_only_ids) - 1, 0, -1):
                if gen_only_ids[i] in [wait_tok, wait_tok_2]:
                    target_idx = prompt_len + i
                    target_token_id = gen_only_ids[i].item()
                    target_label = "Wait"
                    break
            if target_idx is None:
                target_idx = prompt_len + len(gen_only_ids) // 2
                target_token_id = gen_only_ids[target_idx - prompt_len].item()
                target_label = "Middle_Token"

        elif condition in ("normal_thinking", "under_thinking", "term_first"):
            # Find </think> boundary, take the first token after it
            for i in range(len(gen_only_ids) - len(think_end)):
                if gen_only_ids[i : i + len(think_end)].tolist() == think_end:
                    target_idx = prompt_len + i + len(think_end)
                    if target_idx < len(generated_ids):
                        target_token_id = generated_ids[target_idx].item()
                        target_label = "Answer_Start"
                    break
            if target_idx is None:
                target_idx = prompt_len + len(gen_only_ids) - 1
                target_token_id = generated_ids[-1].item()
                target_label = "End_Token"

        target_str = tokenizer.decode([target_token_id]) if target_token_id else "?"
        print(f"Target: layer_pos={target_idx}, label={target_label}, "
              f"token_id={target_token_id}, token_str='{target_str}'")

        # ---- Logit Lens: 29-layer forward pass ----
        input_ids_snap = generated_ids[:target_idx].unsqueeze(0)
        print("Running forward pass for all 29 layers...")
        with torch.no_grad():
            snap_outputs = model(input_ids_snap, output_hidden_states=True)

        hidden_states = snap_outputs.hidden_states
        layer_probs = []

        for layer_idx, h in enumerate(hidden_states):
            last_token_h = h[:, -1, :]
            if layer_idx > 0:
                last_token_h = model.model.norm(last_token_h)
            logits = model.lm_head(last_token_h)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            prob_target = probs[0, target_token_id].item()
            layer_probs.append(prob_target)

        results[condition] = [
            {
                "prompt_prefix": cot_prompt,
                "target_label": target_label,
                "target_token_id": target_token_id,
                "target_token_str": target_str,
                "output_length_chars": len(generated_text),
                "output_tokens": len(gen_only_ids),
                "layer_probs": layer_probs,  # 30 values (layer 0-29)
            }
        ]

        print(f"Layer probs range: [{min(layer_probs):.6f}, {max(layer_probs):.6f}]")

    # ---- Save ----
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "real_logit_lens_data_4conditions.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    for cond, data in results.items():
        d = data[0]
        print(f"  {cond:20s}: len={d['output_tokens']:>6d} tokens, "
              f"target='{d['target_token_str']}', "
              f"peak_prob={max(d['layer_probs']):.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--max-tokens", type=int, default=32000,
                        help="Max new tokens for generation")
    args = parser.parse_args()
    run_extraction(device=args.device, max_new_tokens=args.max_tokens)
