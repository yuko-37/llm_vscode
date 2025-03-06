import time
import os
import requests
from openai import OpenAI
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service


WEB_SITE_URL = "https://cnn.com"

service = Service('/usr/local/bin/chromedriver')
options = webdriver.ChromeOptions()
options.add_argument("headless")
driver = webdriver.Chrome(service=service, options=options)
driver.get(WEB_SITE_URL)

time.sleep(7)

site_response = driver.page_source
driver.quit()

soup = BeautifulSoup(site_response, 'html.parser')
for irrelevant in soup.body(["script", "style", "img", "input"]):
    irrelevant.decompose()

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