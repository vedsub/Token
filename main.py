from google import genai
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(
  api_key="AIzaSyCbUKDTW9e-euCzA86UoPbp-aZm7ZuE77s"
)
response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)