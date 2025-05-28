import os
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from utils.document_loader import confluence_loader
from utils.rag import embedding_document

st.set_page_config(
    page_title="Confluence Document Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 LangChain Chatbot")
st.caption("A basic chatbot using Streamlit and LangChain powered by Gemini Flash 2.5")

load_dotenv()

config = {
    "confluence_api_key": os.getenv("CONFLUENCE_API_KEY"),
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "pinecone_api_key": os.environ.get("PINECONE_API_KEY"),
    "confluence_uri": "https://chutdanai.atlassian.net/wiki/",
    "email": "chutdanai.tho@gmail.com",
    "space": "langchaind",
    "pages": None,
}

documents = confluence_loader(config=config)
embedding_document(config=config, documents=documents)

try:
    chat = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        google_api_key=config.get("gemini_api_key"),
        temperature=0.6,
        verbose=True,
    )
except Exception as e:
    st.error(
        f"Error initializing the language model: {e} \nPlease ensure your GEMINI_KEY is set."
    )
    st.stop()

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(
            content="Hello! I'm your friendly AI assistant. How can I help you today?"
        )
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- Handle User Input ---
user_prompt = st.chat_input("What is up?")

if user_prompt:
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get AI response
    with st.spinner("Thinking..."):
        try:
            response = chat.invoke(st.session_state.messages)
            ai_response_content = response.content
        except Exception as e:
            ai_response_content = f"Sorry, I encountered an error: {e}"

    # Add AI response to session state and display it
    st.session_state.messages.append(AIMessage(content=ai_response_content))
    with st.chat_message("assistant"):
        st.markdown(ai_response_content)

# --- Optional: Clear Chat History Button ---
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [
        AIMessage(
            content="Hello! I'm your friendly AI assistant. How can I help you today?"
        )
    ]
    st.rerun()

st.sidebar.info("This is a basic chatbot. More advanced features can be added!")
