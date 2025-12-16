import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import requests
load_dotenv()

client = OpenAI()


def get_weather(city : str):
  url = f"https://wttr.in/{city.lower()}?format=%C+%t"
  response = requests.get(url)
  if response.status_code == 200:
    return f"The weather in {city} is {response.text}"
  return "Something went wrong while fetching the weather"


SYSTEM_PROMPT = """ 
You are an expert AI assistant using Chain of Thought (CoT) reasoning to help with the user's question.
You MUST respond with a JSON array containing your reasoning steps. Each step must be a JSON object with exactly two fields:
- "step": one of "start", "PLAN", or "output"
- "content": a string with your reasoning or answer
- You can also call a tool , if required from available tools

Format your response as a valid JSON array like this:
[
  {"step": "start", "content": "Understanding what the user wants..."},
  {"step": "PLAN", "content": "My plan to solve this..."},
  {"TOOL" : "str" , "input" : "str"},
  {"step": "output", "content": "The final answer..."}
]
Available Tools :
- get_weather(city : str) : Take name of city as imput and return the weather info of that city

Example 1:
START: What is 25 * 4 + 10?
PLAN: { "step": "PLAN", "content": "User wants to calculate a mathematical expression." }
PLAN: { "step": "PLAN", "content": "I should use the calculator tool to ensure precision." }
PLAN: { "step": "TOOL", "tool": "calculator", "input": "25 * 4 + 10" }
PLAN: { "step": "OBSERVE", "tool": "calculator", "output": "110" }
PLAN: { "step": "PLAN", "content": "Calculation complete. The result is 110." }

Example 2:
START: Write a Python function to check if a number is even.
PLAN: { "step": "PLAN", "content": "User requested a Python function to check if a number is even." }
PLAN: { "step": "PLAN", "content": "I do not need external tools. I will write the logic using the modulo operator." }
PLAN: { "step": "PLAN", "content": "The logic requires checking if the remainder of division by 2 is 0." }
PLAN: { "step": "PLAN", "content": "Constructing function: def is_even(n): return n % 2 == 0" }
PLAN: { "step": "PLAN", "content": "Code generated successfully. Ready to provide output." }
Example 3 :
START: What is the weather of Odisha?
PLAN: { "step": "PLAN", "content": "Seems like user is interested in getting weather of Odisha in India" }
PLAN: { "step": "PLAN", "content": "Lets see if we have any available tool from the list of available tools" }
PLAN: { "step": "PLAN", "content": "Great, we have get_weather tool available for this query." }
PLAN: { "step": "PLAN", "content": "I need to call get_weather tool for odisha as input for city" }
PLAN: { "step": "TOOL", "tool": "get_weather", "input": "odisha" }
PLAN: { "step": "OBSERVE", "tool": "get_weather", "output": "The temp of odisha is clear with 28 C" }
PLAN: { "step": "PLAN", "content": "Great, I got the weather info about odisha" }
PLAN: { "step": "PLAN", "content": "Ready to provide output" }

IMPORTANT: Your entire response must be valid JSON. Do not include any text outside the JSON array.
"""

# Available tools mapping
available_tools = {
    "get_weather": get_weather
}

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
    
    # Agent loop - keeps running until no more tool calls
    while True:
        # Get response from API
        response = client.chat.completions.create(
            model="gpt-4o",
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
            
            # Check for TOOL calls in the response
            tool_called = False
            for parsed_result in parsed_response:
                if parsed_result.get("step") == "TOOL":
                    tool_to_call = parsed_result.get("tool")
                    tool_input = parsed_result.get("input")
                    print(f"🔧: {tool_to_call} ({tool_input})")
                    
                    # Call the tool
                    tool_response = available_tools[tool_to_call](tool_input)
                    
                    # Add tool observation to message history
                    messages.append({
                        "role": "developer",
                        "content": json.dumps({
                            "step": "OBSERVE",
                            "tool": tool_to_call,
                            "input": tool_input,
                            "output": tool_response
                        })
                    })
                    tool_called = True
            
            # If no tool was called, break out of the agent loop
            if not tool_called:
                break
                
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            print(f"\nAssistant: {assistant_reply}")
            break