from modal import App, Image

app = App("testmodal")

@app.function(region="eu")
def func():
    import requests
    
    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data['city'], data['region'], data['country']
    message = f"Hello from {city}, {region}, {country}"
    return message