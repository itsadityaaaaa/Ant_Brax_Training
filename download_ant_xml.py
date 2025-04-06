import requests

url = "https://raw.githubusercontent.com/google/brax/main/brax/envs/assets/ant.xml"

response = requests.get(url)

if response.status_code == 200:
    with open("ant.xml", "wb") as f:
        f.write(response.content)
    print("✅ ant.xml downloaded successfully.")
else:
    print("❌ Failed to download ant.xml")
