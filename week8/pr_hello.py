from modal import App, Image

app = App("test-app")
image = Image.debian_slim().pip_install("requests")

@app.function(image=image)
def hello():
    import requests

    resp = requests.get('https://ipinfo.io/json')
    data = resp.json()
    city, region, country = data['city'], data['region'], data['country']
    return f"Hello from {city}, {region}, {country}!"
        