from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key="AIzaSyCbUKDTW9e-euCzA86UoPbp-aZm7ZuE77s" , 
  base_url = "https://generativelanguage.googleapis.com/v1beta/"
)
response = client.chat.completions.create(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)