
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

model_path = "/home/safiur/Project/models/clintext/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype = torch.float16,
    device_map = "auto",
)

SYSTEM_PROMPT = """
You are a medical AI assistant that rewrites noisy, telegraphic, or poorly formatted clinical text
into clear, grammatically correct, medically faithful prose. Preserve all medical facts and do not
invent new information. Your reply should be only the cleaned clinical text.
"""

raw_content = sys.argv[1]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": raw_content},
]

#from unsloth.chat_templates import get_chat_template
#tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

def format_llama31_chat(messages, add_generation_prompt=True):
    """Manually format messages for Llama 3.1 - lowest resource"""
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    if add_generation_prompt:
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

# Usage: skip tokenizer.chat_template entirely
text = format_llama31_chat(messages)
inputs = tokenizer(text, return_tensors="pt")


outputs = model.generate(
    **inputs,
    max_new_tokens = 256,
    temperature = 0.7,
    top_p = 0.9,
)

# Slice off the input tokens — keep only newly generated ones
generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

corrected_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
).strip()

print(corrected_text)



