# Prerequisites requires ollama to be installed
# and model llama3.2 to have been pulled
# ollama pull llama3.2
 
import ollama


MODEL_LLAMA = 'llama3.2'

system_prompt = "You are a snarky technial assistant and help with code explanation."

def get_user_prompt(code):
    user_prompt = "Please explain what this code does and why: \n" + code
    return user_prompt


def ask_llama(code):
    messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': get_user_prompt(code)}
        ]
    response = ollama.chat(model=MODEL_LLAMA, messages=messages)
    return response['message']['content']


llama_answer = ask_llama('yield from {book.get("author") for book in books if book.get("author")}')
print(llama_answer)