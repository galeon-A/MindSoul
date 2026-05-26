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
    page_title="MindSoul AI | Emotional Wellness Companion",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# AESTHETIC NATURAL CUSTOM CSS
# ---------------------------------------------------
st.markdown(
    '''
    <style>
    /* Main app background - Deep Forest/Sage Gradient */
    .stApp {
        background: linear-gradient(135deg, #14231c 0%, #1e352a 50%, #2d4a3e 100%);
        color: #f4f6f5;
        font-family: 'Inter', sans-serif;
    }

    /* Target headers globally to match aesthetic */
    h1, h2, h3, h4, p {
        color: #eef1f0 !important;
    }

    /* Aesthetic Title Styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #a3b899, #dbe7db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 1rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        text-align: center;
        color: #a7bfae !important;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2.5rem;
    }

    /* Earthy Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 16px;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .feature-tag {
        display: inline-block;
        background: rgba(163, 184, 153, 0.15);
        color: #c2d1be;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 4px;
        border: 1px solid rgba(163, 184, 153, 0.2);
    }

    /* Modern Streamlit Elements Tuning */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 15px !important;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* User chat bubble */
    [data-testid="stChatMessageUser"] {
        background-color: rgba(45, 74, 62, 0.6) !important;
    }
    
    /* Assistant chat bubble */
    [data-testid="stChatMessageAssistant"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    /* Input styling tweaks */
    .stTextInput > div > div > input {
        background-color: #1c2e25 !important;
        color: #eef1f0 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Custom divider line */
    .custom-hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.15), rgba(255,255,255,0));
        margin: 1.5rem 0;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------
st.markdown('<div class="main-title">🌿 MindSoul AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A serene, private space to process thoughts and find emotional balance.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Sanctuary Control")
    st.caption("Configure your privacy-first wellness experience.")
    
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIzaSy..."
    )

    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)

    # Feature List styled with Earthy Tags
    st.markdown("#### ✨ Engine Specs")
    st.markdown(
        '''
        <span class="feature-tag">Gemini 1.5 Flash</span>
        <span class="feature-tag">LangChain Hybrid Routing</span>
        <span class="feature-tag">FAISS Embeddings</span>
        <span class="feature-tag">PubMed Medical API</span>
        <span class="feature-tag">gTTS Voice Engine</span>
        ''', 
        unsafe_allow_html=True
    )

    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)

    # Clean, stylish disclaimer callout box
    st.markdown(
        '''
        <div style="background: rgba(220, 100, 100, 0.1); border-left: 4px solid #dd6b20; padding: 12px; border-radius: 8px; font-size:0.85rem; color:#f6ad55;">
            <strong>Disclaimer:</strong> This app is an AI companion designed for emotional reflection, not a medical device or professional clinical therapy substitute.
        </div>
        ''',
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# INITIALIZE GATEWAY / ROUTING CHECK
# ---------------------------------------------------
if not api_key:
    st.markdown(
        '''
        <div class="glass-card" style="text-align: center; max-width: 600px; margin: 4rem auto;">
            <h3 style="margin-top:0;">Welcome to MindSoul</h3>
            <p style="color: #cbd5e1;">To open your secure emotional wellness room, please input your Google Gemini API Key in the sidebar interface.</p>
        </div>
        ''',
        unsafe_allow_html=True
    )
    st.stop()

# ---------------------------------------------------
# INITIALIZE MODELS
# ---------------------------------------------------
@st.cache_resource
def init_apis(api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.6,
    )
    wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
    pubmed = PubMedAPIWrapper(top_k_results=1, doc_content_chars_max=400)
    return llm, wiki, pubmed

llm, wiki, pubmed = init_apis(api_key)

# ---------------------------------------------------
# CORE LOGIC AGENTS
# ---------------------------------------------------
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.split_documents([Document(page_content=text)])
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_documents(docs, embeddings)


def retrieve_context(query):
    try:
        wiki_data = wiki.run(query)
    except Exception:
        wiki_data = ""

    try:
        pubmed_data = pubmed.run(query)
    except Exception:
        pubmed_data = ""

    combined_text = f"{wiki_data}\n\n{pubmed_data}".strip()

    # Fixed bug: Handle empty context returns explicitly without breaking vectorstore builds
    if not combined_text:
        return "No explicit scientific literature context retrieved."

    try:
        vector_store = create_vector_store(combined_text)
        relevant_docs = vector_store.similarity_search(query, k=2)
        return "\n".join([doc.page_content for doc in relevant_docs])
    except Exception:
        return "No supplementary medical context parsing required."


def generate_response(user_query):
    context = retrieve_context(user_query)

    prompt = f'''
You are an empathetic, gentle, and emotionally supportive AI therapeutic advisor.

Guidelines:
- Keep your tone warm, calm, soothing, and grounded.
- Help the user explore their feelings without judgment.
- Suggest constructive, low-stakes mindfulness or coping strategies.
- Never issue diagnoses or interpret symptoms as specific psychological illness.
- Seamlessly loop in references to the provided medical context only if it is explicitly helpful.
- Gently remind them to speak with professionals if the query implies safety risks.

Context:
{context}

User:
{user_query}

Therapist:
'''
    response = llm.invoke(prompt)
    return response.content


def text_to_speech(text):
    tts = gTTS(text=text, lang="en", slow=False)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    
    with open(temp_audio.name, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

# ---------------------------------------------------
# CHAT SESSION LIFE CYCLE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout wrapper to keep content beautifully spaced
main_container = st.container()

with main_container:
    # Display historical conversation transcript
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User Chat Input interaction field
user_prompt = st.chat_input("Exhale your thoughts here... How are you truly feeling?")

if user_prompt:
    # Append User prompt state
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with main_container:
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # AI Agent Thought-Loop generation
        with st.chat_message("assistant"):
            with st.spinner("Listening deeply..."):
                try:
                    response = generate_response(user_prompt)
                    st.markdown(response)
                    
                    # Modern compact audio presentation
                    audio = text_to_speech(response)
                    st.audio(audio, format="audio/mp3")
                    
                    # Persist response to internal history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Apologies, an unexpected disruption occurred: {str(e)}")
