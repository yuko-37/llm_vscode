import modal
from modal import App, Volume, Image

app = modal.App("llama")
image = Image.debian_slim().pip_install("torch", "bitsandbytes", "transformers", "accelerate")
secrets = [modal.Secret.from_name("hf-secret")]
GPU = 'T4'
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

@app.function(image=image, secrets=secrets, gpu=GPU)
def generate(prompt: str) -> str:
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_type=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto"
    )

    set_seed(37)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=5)
    return tokenizer.decode(outputs[0])