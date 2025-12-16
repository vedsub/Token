import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"), 
  base_url="https://generativelanguage.googleapis.com/v1beta/"
)
SYSTEM_PROMPT = """ 
You are an expert AI assistant using Chain of Thought (CoT) reasoning to help with the user's question.
You MUST respond with a JSON array containing your reasoning steps. Each step must be a JSON object with exactly two fields:
- "step": one of "start", "PLAN", or "output"
- "content": a string with your reasoning or answer

Format your response as a valid JSON array like this:
[
  {"step": "start", "content": "Understanding what the user wants..."},
  {"step": "PLAN", "content": "My plan to solve this..."},
  {"step": "output", "content": "The final answer..."}
]

Example 1:
User: What is 25 * 4 + 10?
Response:
[
  {"step": "start", "content": "The user wants to calculate a mathematical expression."},
  {"step": "PLAN", "content": "First multiply 25 by 4, then add 10 to the result. 25 * 4 = 100, then 100 + 10 = 110"},
  {"step": "output", "content": "The answer is 110."}
]

Example 2:
User: Write a Python function to check if a number is even.
Response:
[
  {"step": "start", "content": "The user wants a Python function to determine if a number is even or odd."},
  {"step": "PLAN", "content": "Create a function that takes a number as input, use the modulo operator to check divisibility by 2, return True if even, False otherwise."},
  {"step": "output", "content": "def is_even(n):\\n    return n % 2 == 0"}
]

IMPORTANT: Your entire response must be valid JSON. Do not include any text outside the JSON array.
"""

# Initialize message history with system prompt
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    # Get user input
    user_input = input("\nYou: ").strip()
    
    # Check for exit conditions
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    if not user_input:
        print("Please enter a question.")
        continue
    
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # Get response from API
    response = client.chat.completions.create(
        model = "gemini-2.0-flash"
,
        messages=messages
    )
    
    # Extract assistant's reply
    assistant_reply = response.choices[0].message.content
    
    # Add assistant's reply to history
    messages.append({"role": "assistant", "content": assistant_reply})
    
    # Print the response in JSON format
    try:
        parsed_response = json.loads(assistant_reply)
        print("\nAssistant:")
        for item in parsed_response:
            print(json.dumps(item, indent=2))
    except json.JSONDecodeError:
        # Fallback if response is not valid JSON
        print(f"\nAssistant: {assistant_reply}")