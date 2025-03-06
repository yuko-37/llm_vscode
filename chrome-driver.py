from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI


service = Service("/usr/local/bin/chromedriver")

options = webdriver.ChromeOptions()
options.add_argument("headless")
# options.add_argument("--no-sandbox")
# options.add_argument("--disable-dev-shm-usage")


driver = webdriver.Chrome(service=service, options=options)
driver.get("https://edwarddonner.com")

time.sleep(5)

page_source = driver.page_source
driver.quit()


soup = BeautifulSoup(page_source, 'html.parser')
text = soup.body.get_text(separator="\n", strip=True)

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI()

system_prompt = "You are a snarky analytic of site content."
user_prompt = "Please analyze the following site text. Respond in markdown\n" + text

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
summary = response.choices[0].message.content

print(summary)