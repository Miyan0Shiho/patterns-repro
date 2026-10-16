import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

target_question = "Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations: \n\\[\\log_2\\left({x \\over yz}\\right) = {1 \\over 2}\\]\n\\[\\log_2\\left({y \\over xz}\\right) = {1 \\over 3}\\]\n\\[\\log_2\\left({z \\over xy}\\right) = {1 \\over 4}\\]\nThen the value of $\\left|\\log_2(x^4y^3z^2)\\right|$ is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.\nPlease put your final answer within \\boxed{}."

# Prompts designed to elicit different behaviors from a 1.5B model
prompts = {
    "under_thinking": [
        "Wait, think."
        target_question
    ],
    "normal_thinking": [
        "Let's reason through this step by step."
        target_question
    ],
    "over_thinking": [
        "Let's think step by step."
        target_question
    ]
}

def get_target_token_id(tokenizer, text):
    # Get the token ID for a specific string (heuristic)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens[0] if len(tokens) > 0 else None

def run_experiment():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading model on MPS...")
    # Load model on MPS in float16 to fit in memory
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="mps"
    )
    model.eval()

    results = {}

    for condition, (cot_prompt, question) in prompts.items():
        print(f"\n--- Processing {condition} (Same Question, Different Prompt) ---")
        full_prompt = f"{cot_prompt}\n\n{question}"
        messages = [{"role": "user", "content": full_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated text length: {len(generated_text)}")
        
        prompt_len = inputs.input_ids.shape[1]
        gen_only_ids = generated_ids[prompt_len:]
        
        target_idx = None
        target_token_id = None
        target_label = ""
            
        # Find token indices
        if condition == "over_thinking":
            # find 'Wait'
            wait_token = get_target_token_id(tokenizer, "Wait")
            wait_token_2 = get_target_token_id(tokenizer, " Wait")
            for i in range(len(gen_only_ids)-1, 0, -1):
                if gen_only_ids[i] in [wait_token, wait_token_2]:
                    target_idx = prompt_len + i
                    target_token_id = gen_only_ids[i].item()
                    target_label = "Wait"
                    break
            if target_idx is None:
                target_idx = prompt_len + len(gen_only_ids) // 2
                target_token_id = gen_only_ids[target_idx - prompt_len].item()
                target_label = "Middle_Token"
                
        elif condition == "normal_thinking":
            # find </think> and take the token right after
            think_end = tokenizer.encode("</think>", add_special_tokens=False)
            for i in range(len(gen_only_ids) - len(think_end)):
                if gen_only_ids[i:i+len(think_end)].tolist() == think_end:
                    target_idx = prompt_len + i + len(think_end)
                    if target_idx < len(generated_ids):
                        target_token_id = generated_ids[target_idx].item()
                        target_label = "Answer_Start"
                    break
            if target_idx is None:
                target_idx = prompt_len + len(gen_only_ids) - 1
                target_token_id = generated_ids[-1].item()
                target_label = "End_Token"
                
        elif condition == "under_thinking":
            # find </think>
            think_end = tokenizer.encode("</think>", add_special_tokens=False)
            for i in range(len(gen_only_ids) - len(think_end)):
                if gen_only_ids[i:i+len(think_end)].tolist() == think_end:
                    target_idx = prompt_len + i + len(think_end)
                    if target_idx < len(generated_ids):
                        target_token_id = generated_ids[target_idx].item()
                        target_label = "Premature_Answer"
                    break
            if target_idx is None:
                target_idx = prompt_len + min(5, len(gen_only_ids)-1)
                target_token_id = generated_ids[target_idx].item()
                target_label = "Early_Token"

        print(f"Target index: {target_idx}, Label: {target_label}, Token ID: {target_token_id}")
        
        # Run forward pass up to target_idx
        input_ids_snap = generated_ids[:target_idx].unsqueeze(0)
        print("Running forward pass for hidden states...")
        with torch.no_grad():
            snap_outputs = model(input_ids_snap, output_hidden_states=True)
            
        hidden_states = snap_outputs.hidden_states # tuple of (batch, seq, dim)
        
        layer_probs = []
        
        # We want the prediction for the NEXT token, which is target_token_id
        for layer_idx, h in enumerate(hidden_states):
            last_token_h = h[:, -1, :] # (1, dim)
            if layer_idx > 0:
                last_token_h = model.model.norm(last_token_h)
            logits = model.lm_head(last_token_h) # (1, vocab_size)
            probs = torch.nn.functional.softmax(logits, dim=-1) # (1, vocab_size)
            prob_target = probs[0, target_token_id].item()
            layer_probs.append(prob_target)
            
        results[condition] = [{
            # Legacy compatibility field for old readers; do not treat it as the formal condition id.
            "prompt_idx": 0,
            "label": target_label,
            "token_id": target_token_id,
            "token_str": tokenizer.decode([target_token_id]),
            "layer_probs": layer_probs
        }]
        
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _DATA = os.path.normpath(os.path.join(_HERE, "..", "data"))
    os.makedirs(_DATA, exist_ok=True)
    out_path = os.path.join(_DATA, "real_logit_lens_data_controlled.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Data saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
