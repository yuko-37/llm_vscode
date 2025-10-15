import modal
from modal import App, Image

app = modal.App("pricer")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")
secrets = [modal.Secret.from_name("hf-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Meta-LLama-3.1-8B"
PROJECT_NAME = 'pricer'
HF_USER = 'yuko-37'
RUN_NAME = "2025-10-02_11.16.17"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

@app.function(image=image, secrets=secrets, gpu=GPU)
def price(description: str) -> float:
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    QUESTION = "How much does this cost to the nearest dollar?"
    PREFIX = "Price is $"

    prompt = f"{QUESTION}\n{description}\n{PREFIX}"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_type=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto"
    )

    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

    set_seed(37)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = fine_tuned_model.generate(**inputs, max_new_tokens=5)
    new_generated = outputs[0][len(inputs.input_ids):]
    text = tokenizer.decode(outputs[0])
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    return float(match.group()) if match else float(0.0)