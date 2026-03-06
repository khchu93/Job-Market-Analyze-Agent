import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_mcp_adapters.client import MultiServerMCPClient  

# LangSmith setup
from langsmith import traceable
from langsmith.wrappers import  wrap_openai

# Generate State Graph
from IPython.display import display, Image 
from pathlib import Path

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")


# llm = init_chat_model("openai:o3-mini")

llm = ChatGoogleGenerativeAI(
    model= "gemini-3-flash-preview",
    temperature=0,
    max_retries=1,
    google_api_key=gemini_api_key
)

# structure output format, add a description of the content
# literal means the return must be exactly the same as one of the list content
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional(therapist) or logical response."
    )


# state what a system has access to
class State(TypedDict):
    messages: Annotated[list, add_messages] # keep track all the previous messages
    message_type: str | None                # track the type of the message (str or None (if empty))
    next: str | None

@traceable(run_type="llm")
def classify_message(state: State):
    last_message = state["messages"][-1]
    # wrap the return to match the desired output format (non text, str classifier here)
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, deals with feellings
            - 'logical': if it asks for facts, information
            """
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])
    return {"message_type": result.message_type}

def router(state: State):
    # default the output to logicial if None
    # get is used to allow a default selection (logicial if none)
    message_type = state.get("message_type", "logical") 
    if message_type == "emotional":
        return {"next": "therapist"}
    else:
        return {"next": "logical"}

@traceable(run_type="llm")
def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

@traceable(run_type="llm")
def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers b ased on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


# create the graph with the State state schema
graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, end_key="classifier")
graph_builder.add_edge(start_key="classifier", end_key="router")

graph_builder.add_conditional_edges(
    source="router",
    path= lambda state: state.get("next"),
    path_map={"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge(start_key="therapist", end_key=END)
graph_builder.add_edge(start_key="logical", end_key=END)

# generate graph
def generate_graph_png(graph, path: str = "agent_graph.png") -> Path:
    png_path = Path(path)
    if not png_path.exists():
        png_bytes = graph.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_bytes)
    return None

# compile graph to something runnable
graph = graph_builder.compile()
# generate graph
generate_graph_png(graph)

def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("bye")
            break
        
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            # print(f"Assistant: {last_message}")
            print(f"Assistant: {last_message.content[0]["text"]}")
            


if __name__ == "__main__":
    run_chatbot()

