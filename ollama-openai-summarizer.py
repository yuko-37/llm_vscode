import requests
from openai import OpenAI
from bs4 import BeautifulSoup

WEB_SITE_URL = "https://edwarddonner.com"

site_response = requests.get(WEB_SITE_URL)
soup = BeautifulSoup(site_response.content, 'html.parser')
site_content = soup.body.get_text(separator='\n', strip=True)

user_prompt = 'Please analyze the following web site content: \n' + site_content

messages = [
    {'role': 'system', 'content': 'You are extremely brief and snarky analytic of web sites.'},
    {"role": "user", "content": user_prompt}
]

ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

response = ollama.chat.completions.create(
    model = 'llama3.2',
    messages=messages
)

print(response.choices[0].message.content)