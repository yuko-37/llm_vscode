from transformers AutoTokenizer, AutoModelForCausalLM, TextStream, BitsAndBytesConfig
from huggingface_hub import login
from google.colab import userdata, drive
import torch
import gc
import speech_recognition as sr
import openai


LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
AUDIO_MODEL = "whisper-1"


audio_filename = 'path-to-audio'

drive.mount('/content/drive/MyDrive/Colab Notebooks')


def transcribe_google(audio_filename):
    text = None
    try:
        recognizer = sr.Recognizer()
        with open(audio_filename) as audio, sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    except Exception as e:
        print(e)
    return text

def transcribe_openai(audio_filename):
    text = None
    with open(audio_filename) as audio:
        text = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio, response_format="text")
    return text


transcript = transcribe_google(audio_filename)


SYSTEM_PROMPT = """You are assistant that produces minutes if meeting 
from transcripts with summary, key discussion points, takeaways and 
action items with owners.
"""


USER_PROMPT = f"""Below is an extract transcript of a Denver council meeting. 
Please write minutes in markdown, including a summary with attendees, 
location and date; discussion points; takeaways; and action items 
with owners.\n{transcript}"""

messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': USER_PROMPT}
]

quant_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_quant_type = 'nf4'
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", 
                                             quantization=quant_config)
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer)

del model, inputs, tokenizer, outputs
gc.collect()
torch.cuda.empty_cache()




