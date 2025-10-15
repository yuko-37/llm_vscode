import modal
from modal import App, Image, Volume

app = App("pricer-service")
image = Image.debian_slim().pip_install("huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft")
secrets = [modal.Secret.from_name("hf-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.1-8B"
HF_USER = "yuko-37"
FINETUNED_MODEL = f"{HF_USER}/pricer-2025-10-02_11.16.17"
CACHE_DIR = "/cache"

MIN_CONTAINERS = 0

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets, 
    gpu=GPU, 
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume}
)
class Pricer():
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            quantization_config=quant_config,
            device_map="auto"
        )
        self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, FINETUNED_MODEL)

    @modal.method()
    def price(self, description: str) -> float:
        import re
        from transformers import set_seed
    
        prompt = f"{QUESTION}\n{description}\n{PREFIX}"
        set_seed(37)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.fine_tuned_model.generate(**inputs, max_new_tokens=5)
        new_generated = outputs[0][len(inputs.input_ids):]
        text = self.tokenizer.decode(outputs[0])
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else float(0.0)