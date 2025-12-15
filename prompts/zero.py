import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"), 
  base_url="https://generativelanguage.googleapis.com/v1beta/"
)
SYSTEM_PROMPT = "You should only answer coding related questions and your name is ALexa , If asked something else, just say sorry." 
response = client.chat.completions.create(
  model ="gemini-2.5-flash",
  messages=[
    {"role":"system", "content":SYSTEM_PROMPT} ,
    {"role":"user", "content":"code to translate english to hindi using python"}
    ]
)

print(response.choices[0].message.content)