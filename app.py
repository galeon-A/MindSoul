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
    page_icon="🌿",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM AESTHETIC THEMING (Natural Zen Colors)
# ---------------------------------------------------
st.markdown(
    '''
    <style>
    /* Main App Background & Text */
    .stApp {
        background-color: #f4f6f4; /* Soft Sage/Alabaster tint */
        color: #2c3e35; /* Deep slate green for high readability */
    }

    /* Sidebar Contrast Styling */
    [data-testid="stSidebar"] {
        background-color: #1e2925 !important; /* Deep Forest/Charcoal Contrast */
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important; /* Crisp, readable text inside sidebar */
    }
    [data-testid="stSidebar"] input {
        color: #1e2925 !important; /* Dark text inside the API input field */
    }

    /* Header Typo */
    .main-title {
        font-family: 'Helvetica Neue', Inter, sans-serif;
        font-size: 2.75rem;
        font-weight: 700;
        text-align: center;
        color: #1e2925;
        margin-top: 1rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-family: 'Helvetica Neue', Inter, sans-serif;
        text-align: center;
        color: #5c6f65;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Clean Info Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }

    /* Target standard Streamlit Chat message styles softly */
    .stChatMessage {
        background-color: #ffffff !important;
        border: 1px solid #e1e8e4 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Target user message variation */
    [data-testid="stChatMessageUser"] {
        background-color: #e8efe9 !important; /* Gentle green tint for user */
        border: 1px solid #d4dfd6 !important;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="main-title">🌿 MindSoul AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A calm, supportive space for your emotional wellness</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR (High Contrast)
# ---------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='margin-top:0;'>⚙️ Workspace Setup</h2>", unsafe_allow_html=True)

    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIzaSy..."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '''
        <div class="feature-card">
            <h4 style="margin-top:0; color:#fff;">✨ Capabilities</h4>
            <p style="font-size:0.9rem; line-height:1.4; margin:0;">
                • Empathetic Chat Companion<br>
                • PubMed Medical Insights<br>
                • Wikipedia Reference Engine<br>
                • Natural Voice Synthesis<br>
                • RAG-driven Context (FAISS)
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.caption(
        "⚠️ Disclaimer: MindSoul is an AI companion designed for reflection, not a medical substitute for clinical therapy."
    )

# ---------------------------------------------------
# CHECK API KEY
# ---------------------------------------------------
if not api_key:
    st.info("👋 Welcome! Please provide your Gemini API key in the sidebar configuration to begin your session.")
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
# USER INPUT & RUNTIME
# ---------------------------------------------------
user_prompt = st.chat_input(
    "What is on your mind today?"
)

if user_prompt:
    # Append User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process & Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Listening mindfully..."):
            try:
                response = generate_response(user_prompt)
                st.markdown(response)

                # Audio generation
                audio = text_to_speech(response)
                st.audio(audio, format="audio/mp3")

                # Save Response State
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
