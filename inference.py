from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Paths
adapter_path = "DAY - 40 Nividia RLHF\\tinyllama-lora-sft-tuned-model"  # LoRA adapter SFT
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"                  # Base model
dpo_model_path = "DAY - 40 Nividia RLHF/tinyllama-dpo-merged-final"      # DPO merged full model

# Prepare prompt exactly as training
instruction = "What is the primary focus of STAC-A2 benchmarks?"
prompt = f"According to the following question\n\n{instruction}\nAnswer:\n\n"

# Tokenize prompt once
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
inputs = tokenizer(prompt, return_tensors="pt")

# 1. Load base + LoRA SFT adapter model and generate
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model_sft = PeftModel.from_pretrained(base_model, adapter_path)

print("Generating output from base model (no LoRA):")
base_outputs = base_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(base_outputs[0], skip_special_tokens=True))

print("\nGenerating output from LoRA SFT tuned model:")
sft_outputs = model_sft.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(sft_outputs[0], skip_special_tokens=True))

# 2. Load merged DPO model and generate
tokenizer_dpo = AutoTokenizer.from_pretrained(dpo_model_path)
model_dpo = AutoModelForCausalLM.from_pretrained(dpo_model_path)

inputs_dpo = tokenizer_dpo(prompt, return_tensors="pt")

print("\nGenerating output from merged DPO tuned model:")
dpo_outputs = model_dpo.generate(**inputs_dpo, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer_dpo.decode(dpo_outputs[0], skip_special_tokens=True))
