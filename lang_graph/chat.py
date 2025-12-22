import os
from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

# Initialize HuggingFace LLM (free tier with HF token)
hf_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
    max_new_tokens=512,
    temperature=0.7
)
llm = ChatHuggingFace(llm=hf_llm)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str  # Track which route was taken

# Router function - determines which node to go to based on message content
def route_message(state: State) -> Literal["help_node", "joke_node", "chatbot"]:
    """Analyze the last message and route to appropriate node."""
    last_message = state["messages"][-1]
    content = last_message.content.lower() if hasattr(last_message, 'content') else str(last_message).lower()
    
    if "help" in content or "?" in content:
        return "help_node"
    elif "joke" in content:
        return "joke_node"
    else:
        return "chatbot"

# Node: Help handler
def help_node(state: State):
    print("\n🆘 Routed to: HELP NODE")
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant. Provide clear, concise help."},
        *state["messages"]
    ])
    return {"messages": [response], "route": "help"}

# Node: Joke handler
def joke_node(state: State):
    print("\n😂 Routed to: JOKE NODE")
    response = llm.invoke([
        {"role": "system", "content": "You are a comedian. Tell a short, funny joke related to the topic."},
        *state["messages"]
    ])
    return {"messages": [response], "route": "joke"}

# Node: General chatbot
def chatbot(state: State):
    print("\n💬 Routed to: CHATBOT NODE")
    response = llm.invoke(state["messages"])
    return {"messages": [response], "route": "chat"}

# Build the graph with conditional edges
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("help_node", help_node)
graph_builder.add_node("joke_node", joke_node)
graph_builder.add_node("chatbot", chatbot)

# Add conditional edge from START based on route_message
graph_builder.add_conditional_edges(
    START,
    route_message,
    {
        "help_node": "help_node",
        "joke_node": "joke_node",
        "chatbot": "chatbot"
    }
)

# All nodes go to END
graph_builder.add_edge("help_node", END)
graph_builder.add_edge("joke_node", END)
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Test with different inputs
if __name__ == "__main__":
    print("=" * 50)
    print("Testing LangGraph Conditional Edges")
    print("=" * 50)
    
    # Test 1: Help request
    print("\n📝 Test 1: 'I need help with Python'")
    result = graph.invoke({"messages": ["I need help with Python"], "route": ""})
    print(f"Response: {result['messages'][-1].content[:200]}...")
    
    # Test 2: Joke request  
    print("\n📝 Test 2: 'Tell me a joke about programming'")
    result = graph.invoke({"messages": ["Tell me a joke about programming"], "route": ""})
    print(f"Response: {result['messages'][-1].content[:200]}...")
    
    # Test 3: Regular chat
    print("\n📝 Test 3: 'Hello, my name is Vedant'")
    result = graph.invoke({"messages": ["Hello, my name is Vedant"], "route": ""})
    print(f"Response: {result['messages'][-1].content[:200]}...")