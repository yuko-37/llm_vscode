from website import Website

def get_links_system_prompt():
    prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
    prompt += "You should respond in JSON as in this example:"
    prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""
    return prompt


def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt


def get_brochure_system_prompt():
    prompt = 'You are a snarky assistant. You make funny jokey \
short brochures abount company.'
    return prompt


def get_brochure_user_prompt(company, details):
    prompt = f'Please create a short brochure about company {company} \
using your knowledge and further information.\n'
    prompt += details
    return prompt