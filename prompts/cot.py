import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"), 
  base_url="https://generativelanguage.googleapis.com/v1beta/"
)
SYSTEM_PROMPT = """ 
You are an expert AI assistant using Chain of Thought (CoT) reasoning to help with the user's question.
Use the Start, Plan, Action, Output format. Give output only after the final plan has been made.

Example 1:
User: What is 25 * 4 + 10?
Start: The user wants to calculate a mathematical expression.
Plan: First multiply 25 by 4, then add 10 to the result.
Action: 25 * 4 = 100, then 100 + 10 = 110
Output: The answer is 110.

Example 2:
User: Write a Python function to check if a number is even.
Start: The user wants a Python function to determine if a number is even or odd.
Plan: Create a function that takes a number as input, use the modulo operator to check divisibility by 2, return True if even, False otherwise.
Action: Define function with parameter, use n % 2 == 0 condition.
Output:
def is_even(n):
    return n % 2 == 0

Now solve the user's question using the same format.
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
    
    # Print the response
    print(f"\nAssistant: {assistant_reply}")