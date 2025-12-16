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

def main():
    user_query = input("> ")
    # Initialize messages with system prompt (reuse from agent if needed)
    messages = [{"role": "user", "content": user_query}]
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        assistant_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_reply})
        try:
            parsed = json.loads(assistant_reply)
            print("\nAssistant:")
            for item in parsed:
                print(json.dumps(item, indent=2))
            tool_called = False
            for step in parsed:
                if step.get("step") == "TOOL":
                    tool_name = step.get("tool")
                    tool_input = step.get("input")
                    print(f"🔧 Calling {tool_name}({tool_input})")
                    if tool_name == "get_weather":
                        tool_output = get_weather(tool_input)
                    else:
                        tool_output = f"Tool {tool_name} not implemented"
                    # Append observation
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({
                            "step": "OBSERVE",
                            "tool": tool_name,
                            "input": tool_input,
                            "output": tool_output
                        })
                    })
                    tool_called = True
            if not tool_called:
                break
        except json.JSONDecodeError:
            print(f"\nAssistant: {assistant_reply}")
            break

if __name__ == "__main__":
    main()