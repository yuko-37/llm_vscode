import re

from datasets import load_dataset
from typing import Optional
from transformers import AutoTokenizer


BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

MIN_TOKENS = 150 # Any less than this, and we don't have enough useful content
MAX_TOKENS = 160 # Truncate after this many tokens. Then after adding in prompt text, we will get to around 180 tokens

MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 5

SAFE_ESCAPE = """[\\\\!"\\\#\\\$%\\\&'\\\(\\\)\\\*\\\+,\\\-\\\./:;<=>\\\?@\\\[\\\\\]\\\^`\\\{\\\|\\\}\\\~¡£¤¥¦§¨©«¬­®¯°±´·¸»¿×˚́̊​‍‎‏‐‑‒–—―‘’‚“”„•…‰′″›※‼‿⁄€⃣℃℉™↑→↖↘↙∕√∞≈≤≥≦≪⌀⌛⌦⏩⏱⏳⏺ⒶⒷⒸⒹⒺⓇ■▣▪▲△▶▷▸►▼▽◀◄◆◇◈◉◎●◕◢◤◥◦◼◾☀☁☃★☆☉☎☑☔☕☛☝☞☪☰☺☻♀♂♐♕♚♛♠♡♣♥♦♨♪♫♬♻⚒⚗⚙⚠⚡⚪⚫⛅⛓⛔⛳✂✅✆✈✉✊✋✌✍✎✏✐✓✔✚✤✦✧✨✩✪✫✮✯✸✽✾✿❀❁❃❄❅❆❇❉❋❌❎❓❖❗❛❜❣❤❥❰❱➕➡➢➤➥➬➯➱➼➽⬛⬜⭐⭕、。〃《》「」『』【】〔〕〖〗〚〛〜・㋡㎛㎡︎️！＂＃＄％＆（）＋，－．／：；＜＞？［］｜～｡･�]"""
ESCAPE_PATTERN = re.compile(SAFE_ESCAPE, flags=re.UNICODE)


class Item:
    """
    An Item is a cleaned, curated datapoint of a Product with a Price
    """
    _tokenizer = None
 
    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        return cls._tokenizer

    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    REMOVALS = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', 
                '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', 
                "By Manufacturer", "Item", "Date First", "Package", ":", "Number of",
                "Best Sellers", "Number", "Product "]
    
    def __init__(self, datapoint, price, category=''):
        self.content: str = ''
        self.category: str = category
        self.token_count: int = 0
        self.details: Optional[str] = None
        self.prompt: Optional[str] = None
        self.include = False
        
        self.title = datapoint['title']
        self.price = price
        self.parse(datapoint)
        
    def parse(self, stuff):
        descr = stuff['description']
        features = stuff['features']
        self.details = stuff['details']
        
        content = ""
        if descr:
            content += "\n".join(descr) + "\n"
        if features:
            content += "\n".join(features) + "\n"
        if self.details:
            content += self.scrub_details() + "\n"

        self.raw_content = content
        self.content = self.scrub(content)
        
        if len(self.content) > MIN_CHARS:
            self.content = self.content[:CEILING_CHARS]
            text =f"{self.title}\n{self.content}"
            tokenizer = self.get_tokenizer()
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include=True
                
    def scrub_details(self):
        details = self.details
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details
    
    def scrub(self, stuff):
        stuff = ESCAPE_PATTERN.sub('', stuff).strip()
        words = stuff.split(" ")
        filtered = [w for w in words if len(w) < 7 or not any(char.isdigit() for char in w)]
        return " ".join(filtered)

    def make_prompt(self, text):
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{self.price:.2f}"
        self.token_count = len(self.get_tokenizer().encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX
        
    def __repr__(self):
        return (f"Title: {self.title}\nPrice: {self.price:.2f}\nInclude: {self.include} {self.token_count} tokens"
               f"\nContent:\n{self.content}")
