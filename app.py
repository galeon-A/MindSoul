
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import WikipediaAPIWrapper, PubMedAPIWrapper
from langchain.docstore.document import Document

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_GEMINI_API_KEY in .env file")
    st.stop()

# -----------------------------
# Configure Streamlit
# -----------------------------
st.set_page_config(
    page_title="MindMate AI Therapist",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Gemini LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
)

# -----------------------------
# Data Sources
# -----------------------------
wiki = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=500
)

pubmed = PubMedAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=500
)

# -----------------------------
# Utility Functions
# -----------------------------
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = splitter.split_documents([Document(page_content=text)])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def retrieve_context(query):
    try:
        wiki_data = wiki.run(query)
    except Exception:
        wiki_data = ""

    try:
        pubmed_data = pubmed.run(query)
    except Exception:
        pubmed_data = ""

    combined_text = f"{wiki_data}\n\n{pubmed_data}"

    if not combined_text.strip():
        return "No external context available."

    vector_store = create_vector_store(combined_text)
    relevant_docs = vector_store.similarity_search(query, k=3)

    return "\n".join([doc.page_content for doc in relevant_docs])


def generate_response(user_query):
    context = retrieve_context(user_query)

    prompt = f'''
You are an empathetic and supportive AI therapist.

Guidelines:
- Be compassionate and emotionally supportive.
- Never diagnose serious medical conditions.
- Suggest healthy coping mechanisms.
- Encourage professional help for severe mental health concerns.
- Keep responses warm and conversational.

Context:
{context}

User:
{user_query}

Therapist:
'''

    response = llm.invoke(prompt)
    return response.content


def text_to_speech(text):
    tts = gTTS(text=text, lang="en")

    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    tts.save(temp_audio.name)

    with open(temp_audio.name, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return audio_bytes


# -----------------------------
# UI
# -----------------------------
st.title("🧠 MindMate AI Therapist")
st.caption("Your AI-powered mental wellness companion")

with st.sidebar:
    st.header("About")
    st.write(
        '''
MindMate AI Therapist uses:
- Google Gemini
- LangChain
- Wikipedia
- PubMed
- FAISS Vector Search

⚠️ This app is not a replacement for professional therapy.
        '''
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("How are you feeling today?")

if user_prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(user_prompt)

            st.markdown(response)

            audio = text_to_speech(response)
            st.audio(audio, format="audio/mp3")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
