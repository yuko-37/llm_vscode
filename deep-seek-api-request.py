from openai import OpenAI

ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

response = ollama.chat.completions.create(
    model="deepseek-r1:1.5b",
    messages=[{"role": "user", "content": "Please give definitions of some core concepts behind LLMs: a neural network, attention and the transformer"}]
)

print(response.choices[0].message.content)