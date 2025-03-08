import os
import json
import time
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from tools import TavilySearchTool, FlightSearchTool, CompanyVectorStore, CompanySearchTool
from core import tracer


# Initialize CompanyVectorStore and CompanySearchTool globally
search_tool = TavilySearchTool(max_results=1, include_answer=True) # include_raw_content=True
flight_tool = FlightSearchTool()
tools = [flight_tool, search_tool] 

if "company_vectorstore" not in st.session_state:
    st.session_state.company_vectorstore = CompanyVectorStore()
company_vectorstore = st.session_state.company_vectorstore
company_search_tool = CompanySearchTool(company_vectorstore=company_vectorstore) 

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)
llm_with_tools = llm.bind_tools(tools)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.placeholder = container.empty()
    
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.placeholder.markdown(self.text)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    sources: List[Dict[str, Any]] # To hold the sources with metadata

def format_context_with_citations(docs):
        context = ""
        for idx, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "Unknown page")
            context += f"[{idx + 1}] (File: {source}, Page: {page})\n{doc.page_content}\n\n"
        return context

@tracer.agent(name="company_search_node")
def company_search_node(state: State):
    past_messages = state.get("messages", [])
    current_sources = state.get("sources", [])
    
    user_query = next(
        (msg.content for msg in reversed(past_messages) if isinstance(msg, HumanMessage)),
        None
    )

    if not user_query:
        raise ValueError("No human message found for processing.")

    # Retrieve documents
    results = company_search_tool.invoke(user_query)

    if not results:
        new_message = AIMessage(content="No relevant information found in Rakuten's financial documents.")
        return {
            "messages": past_messages + [new_message],
            "sources": current_sources,
        }
    
    citation_prompt_template: str = """
        You are a financial analyst. Use the following excerpts from Rakuten's financial documents to answer the user's question.

        When using information from the context, include inline citations like [1], [2] that correspond to the provided excerpts.

        Question:
        {question}

        Context:
        {context}

        Answer:
    """

    context = format_context_with_citations(results)
    prompt = citation_prompt_template.format(question=user_query, context=context)

    search_llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)
    search_llm_response = search_llm.invoke(prompt)

    sources = [
        {
            "source": doc.metadata.get("source", "Unknown source"),
            "page": doc.metadata.get("page", "Unknown page"),
            "content": doc.page_content.strip(),
        }
        for doc in results
    ]

    all_sources = current_sources + sources

    new_message = AIMessage(content=search_llm_response.content)

    return {
        "messages": past_messages + [new_message],
        "sources": all_sources,
    }

def format_message_for_logging(message):
    """Helper function to format message objects for readable logging"""
    if isinstance(message, dict):
        return message
    
    # Handle AIMessage, HumanMessage, or other message types
    return {
        "content": message.content if hasattr(message, "content") else str(message),
        "type": message.__class__.__name__,
        "additional_kwargs": message.additional_kwargs if hasattr(message, "additional_kwargs") else {},
    }

def format_response_for_logging(response):
    """Format the entire response for readable logging"""
    return {
        "messages": [format_message_for_logging(msg) for msg in response["messages"]],
        "sources": response.get("sources", [])
    }

@tracer.chain(name="is_rakuten_query")
def is_rakuten_query(messages):
    # Get the last message content, handling both dict and Message objects
    last_message = messages[-1]
    if isinstance(last_message, dict):
        content = last_message["content"]
    else:
        # Handle LangChain message objects
        content = last_message.content
    
    return "rakuten" in content.lower()

@tracer.chain(name="router")
def router(state):
    if tools_condition(state) == "tools":
        return "tools"
    if state.get("next") == "company_search":
        return "company_search"
    return "end"

@tracer.agent(name="chatbot")
def chatbot(state: State):
    past_messages = state.get("messages", [])
    current_sources = state.get("sources", [])

    if is_rakuten_query(past_messages):
        return {"next": "company_search", "messages": past_messages, "sources": current_sources} # Prevent LLM response, route immediately

    response = llm_with_tools.invoke(past_messages)

    print(f"[LOG] Chatbot response type: {type(response)}")
    print(f"\n\n****[LOG] Chatbot response: {response}\n\n")
    
    # Append LLM response to memory
    return {"messages": past_messages + [response], "sources": current_sources}

def show_example_queries():
    """Display example queries in the sidebar"""
    st.markdown("""
    ##### 🛫 Flight Search
    - Show me flights from Bangalore to Tokyo
    - Find flights from NYC to London next week
    - What are the available flights from Paris to Berlin?
    
    ##### 📰 News Search
    - Show me latest news about cricket
    - What's happening in AI technology today?
    - Tell me recent news about SpaceX
    
    ##### 💹 Company Search
    - Show me financial performance of Rakuten for 2024
    - What was Rakuten's revenue in Q3 2023?
    - Tell me about Rakuten's mobile business performance
    """)

def main():

    # Set page config
    st.set_page_config(page_title="Multi-Agentic Assistant: Flight Search, News, and Company Financials", layout="wide")

    # Add CSS to make the chat input stick to bottom
    st.markdown("""
        <style>
        .stChatFloatingInputContainer, .stChatInput {
            position: fixed;
            bottom: 0;
            right: 0;
            width: 75%;  /* Matches the right column width */
            padding: 1rem;
            z-index: 100;
        }
        
        </style>
        """, unsafe_allow_html=True)
    
    # Create two columns with 25-75 split
    left_col, right_col = st.columns([1, 3])

    # Left column: Example queries
    with left_col:
        st.markdown("### Multi-Agentic Assistant")
        show_example_queries()
        log_container = st.expander("📝 Debug Logs", expanded=False)

    with right_col:

        # Create containers for chat layout
        chat_container = st.container()

        memory = MemorySaver()

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("company_search", company_search_node)
        graph_builder.add_node("tools", ToolNode(tools=tools))

        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_conditional_edges("chatbot", router)
        graph_builder.add_conditional_edges(
            "company_search",
            lambda state: "chatbot" if state["messages"][-1].content == "No relevant information found in Rakuten's financial documents." else "end"
        )

        graph_builder.set_entry_point("chatbot")
        graph = graph_builder.compile(checkpointer=memory)

        # Initialize session state for UI
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = "chat_" + str(int(time.time()))
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Get user input
        user_input = st.chat_input("Type your message here...")

        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_input)

                # Get AI response
                with st.chat_message("assistant"):
                    stream_container = st.container()
                    stream_handler = StreamHandler(stream_container)
                    
                    # Invoke the graph
                    response = graph.invoke(
                        {"messages": st.session_state.messages, "sources": []},
                        {
                            "configurable": {"thread_id": st.session_state.thread_id},
                            "callbacks": [stream_handler],
                        }
                    )

                    formatted_response = format_response_for_logging(response)
                    print("Response from graph:", json.dumps(formatted_response, indent=2))

                    with log_container:
                        st.code(json.dumps(formatted_response, indent=2), language="json")
                    
                    # Add assistant response to chat history
                    ai_message = response["messages"][-1]
                    
                    # Persist Memory
                    st.session_state.messages.append({"role": "assistant", "content": ai_message.content})

                    # Display citations if available from the CompanySearchTool
                    if response.get("sources"):
                        st.markdown("##### Sources:")
                        for idx, doc in enumerate(response["sources"]):
                            with st.expander(f"[{idx + 1}] {doc['source']}, Page {doc['page']}"):
                                st.write(doc["content"])
        
       


if __name__ == "__main__":
    main()