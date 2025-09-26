import os
import json
import aiprompts as p
from website import Website
from openai import OpenAI
from dotenv import load_dotenv
from utils import log, clear_logs


MODEL = 'gpt-4o-mini'
openai = OpenAI()


def get_relevant_links_as_json(website):
    response = openai.chat.completions.create(
        model = MODEL,
        messages=[
            {"role": "system", "content": p.get_links_system_prompt()},
            {"role": 'user', 'content': p.get_links_user_prompt(website)}
        ],
        response_format={'type': 'json_object'}
    )
    result = response.choices[0].message.content
    log(result, label='Links result:')
    return json.loads(result)


def get_all_details(website: Website):
    details = f'Landing page:\n {website.get_contents}'
    links = get_relevant_links_as_json(website)
    
    for link in links["links"]:
        details += f"\n\n{link['type']}:\n"
        embedded = Website(link["url"])
        details += embedded.get_contents()
    return details[:5_000]


def create_brochure(s_prompt, u_prompt):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', "content": s_prompt},
            {'role': 'user', "content": u_prompt}
        ]
    )
    result = response.choices[0].message.content
    return result


def print_brochure_from_stream(s_prompt, u_promt):
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', "content": s_prompt},
            {'role': 'user', "content": u_prompt}
        ],
        stream=True
    )

    text = ''
    for chunk in stream:
        text += chunk.choices[0].delta.content or ''
    print(text)

os.system('clear')
print('start...')
clear_logs(['out.log', 'brochure.md'])
load_dotenv()

company = 'Anthropic'
url = "https://www.anthropic.com"
website = Website(url)

log(p.get_links_system_prompt(), "Links System prompt:")
log(p.get_links_user_prompt(website), "Links User prompt:")

details = get_all_details(website)
s_prompt = p.get_brochure_system_prompt()
u_prompt = p.get_brochure_user_prompt(company, details)

log(s_prompt, 'Brochure System prompt: ')
log(u_prompt, 'Brochure User prompt: ')

brochure = create_brochure(s_prompt, u_prompt)
log(brochure, filename='brochure.md')

# print_brochure_from_stream(s_prompt, u_prompt)
print('done')


