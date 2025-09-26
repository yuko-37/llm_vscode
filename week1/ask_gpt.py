from dotenv import load_dotenv
from openai import OpenAI

MODEL_GPT = 'gpt-4o-mini'

load_dotenv()
openai = OpenAI()

system_prompt = "You are a snarky technial assistant and help with code explanation. \
Respond in plain text."

def get_user_prompt(code):
    user_prompt = "Please explain what this code does and why: \n" + code
    return user_prompt


def ask_gpt(code):
    response = openai.chat.completions.create(
        model=MODEL_GPT,
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': get_user_prompt(code)}
        ]
    )
    
    return response.choices[0].message.content


def ask_gpt_stream(code):
    stream = openai.chat.completions.create(
        model = MODEL_GPT,
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': get_user_prompt(code)}
        ],
        stream=True
    )
    text = ""
    
    for chunk in stream:
        text += chunk.choices[0].delta.content or ''

    return text


# gpt_answer = ask_gpt('yield from {book.get("author") for book in books if book.get("author")}')
# print(gpt_answer)

gpt_answer = ask_gpt_stream('display(Markdown(text))')
print(gpt_answer)