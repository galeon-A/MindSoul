
import streamlit as st
import tempfile
from gtts import gTTS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import WikipediaAPIWrapper, PubMedAPIWrapper
from langchain.docstore.document import Document

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="MindSoul AI Therapist",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown(
    '''
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #0f172a, #1e293b);
        color: white;
    }

    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .glass-box {
        background: rgba(255,255,255,0.08);
        padding: 1rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }

    .stChatMessage {
        border-radius: 18px;
        padding: 10px;
    }

    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: white;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown('<div class="main-title">🧠 MindSoul AI Therapist</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Your AI-powered emotional wellness companion</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Enter Google Gemini API Key",
        type="password"
    )

    st.markdown("---")

    st.markdown(
        '''
### ✨ Features
- AI Therapist Chat
- PubMed Knowledge
- Wikipedia Context
- Voice Responses
- Gemini AI
- LangChain + FAISS
        '''
    )

    st.markdown("---")

    st.warning(
        "⚠️ This app is not a substitute for professional therapy."
    )

# ---------------------------------------------------
# CHECK API KEY
# ---------------------------------------------------

if not api_key:
    st.info("Please enter your Gemini API key in the sidebar.")
    st.stop()

# ---------------------------------------------------
# INITIALIZE MODELS
# ---------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.7,
)

wiki = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=500
)

pubmed = PubMedAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=500
)

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------

def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = splitter.split_documents(
        [Document(page_content=text)]
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    vector_store = FAISS.from_documents(
        docs,
        embeddings
    )

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

    relevant_docs = vector_store.similarity_search(
        query,
        k=3
    )

    return "\n".join(
        [doc.page_content for doc in relevant_docs]
    )


def generate_response(user_query):

    context = retrieve_context(user_query)

    prompt = f'''
You are an empathetic and emotionally supportive AI therapist.

Guidelines:
- Be warm, calm, and compassionate.
- Help users process emotions.
- Suggest healthy coping mechanisms.
- Never diagnose serious mental illnesses.
- Encourage professional help if needed.
- Keep responses supportive and conversational.

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

# ---------------------------------------------------
# CHAT MEMORY
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# DISPLAY CHAT
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

user_prompt = st.chat_input(
    "How are you feeling today?"
)

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

        with st.spinner("MindSoul is thinking..."):

            try:
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

            except Exception as e:
                st.error(f"Error: {str(e)}")
