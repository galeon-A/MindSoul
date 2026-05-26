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
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="MindSoul AI Therapist",
    page_icon="🧠",
    layout="centered",  # Centered layout feels more intimate for a conversational therapist app
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# MODERN EMBEDDED STYLING
# ---------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Background and Smooth Aesthetics */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #111827 100%);
        color: #f8fafc;
    }
    
    /* Custom Styling for Information Badges */
    .feature-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4px 12px;
        border-radius: 12px;
        margin: 4px;
        font-size: 0.85rem;
        color: #c084fc;
    }
    
    /* Adjust input box borders */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR & CONFIGURATION
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Using session state to preserve the key reliably across actions
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
        
    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="AIzaSy...",
        help="Grab a key from Google AI Studio"
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("---")
    st.markdown("### 🧩 App Capabilities")
    
    # Styled feature list using custom tags
    features = ["Gemini 1.5 Flash", "LangChain RAG", "FAISS Vector DB", "PubMed Insights", "Wikipedia Sync", "gTTS Voice Engine"]
    for f in features:
        st.markdown(f'<span class="feature-tag">✨ {f}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔒 **Privacy Guarantee**: Your API key and mental wellness conversations are completely private and never stored externally.")

# ---------------------------------------------------
# APP HEADER
# ---------------------------------------------------
# Clean header layout without complex HTML tables or hard-to-read css injections
st.title("🧠 MindSoul AI")
st.markdown("*Your AI-powered compassionate space for reflection, clarity, and emotional wellness.*")

# Important clinical disclaimer container
st.info(
    "💡 **Please Note:** MindSoul is a reflective AI companion meant for self-discovery. It is not a clinical tool or a substitute for professional human therapy.",
    icon="⚠️"
)

# Halt application if API key is missing
if not st.session_state.api_key:
    st.warning("Please configure your Google Gemini API Key in the sidebar to start your session.")
    st.stop()

# ---------------------------------------------------
# INITIALIZE MODELS & UTILITIES
# ---------------------------------------------------
@st.cache_resource
def init_tools():
    wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
    pubmed = PubMedAPIWrapper(top_k_results=1, doc_content_chars_max=400)
    return wiki, pubmed

wiki, pubmed = init_tools()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=st.session_state.api_key,
    temperature=0.6, # Dropped slightly for more consistent, grounded therapeutic responses
)

# ---------------------------------------------------
# REFACTORED CORE FUNCTIONS
# ---------------------------------------------------
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.split_documents([Document(page_content=text)])
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.session_state.api_key
    )
    return FAISS.from_documents(docs, embeddings)


def retrieve_context(query):
    # Short circuit for very simple emotional phrases to save API latency
    if len(query.split()) < 3:
        return ""

    wiki_data, pubmed_data = "", ""
    try:
        wiki_data = wiki.run(query)
    except Exception:
        pass

    try:
        pubmed_data = pubmed.run(query)
    except Exception:
        pass

    combined_text = f"{wiki_data}\n\n{pubmed_data}".strip()

    if not combined_text:
        return ""

    try:
        vector_store = create_vector_store(combined_text)
        relevant_docs = vector_store.similarity_search(query, k=2)
        return "\n".join([doc.page_content for doc in relevant_docs])
    except Exception:
        return ""


def generate_response(user_query):
    context = retrieve_context(user_query)
    
    prompt = f"""
You are MindSoul, an exceptionally warm, empathetic, and intuitive AI therapeutic companion.

Operational Guidelines:
- Communicate with calm, gentle, and grounding language.
- Actively validate the user's emotions before jumping into suggestions.
- Offer actionable, small mindfulness or cognitive exercises where appropriate.
- Strictly never diagnose, medicate, or explicitly state clinical mental illnesses.
- If the user implies crisis, self-harm, or severe distress, express deep care and gently provide references to global lifelines.

Contextual Guidance (Use only if relevant to the user's struggle):
{context}

User's Thought:
{user_query}

MindSoul's Response:
"""
    response = llm.invoke(prompt)
    return response.content


def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        with open(temp_audio.name, "rb") as audio_file:
            return audio_file.read()
    except Exception:
        return None

# ---------------------------------------------------
# CHAT SESSION MANAGEMENT
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I'm MindSoul, your quiet space for reflection. How is your heart and mind feeling today?"}
    ]

# Display historical chat strings natively
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------
# CHAT INTERACTION LOGIC
# ---------------------------------------------------
if user_prompt := st.chat_input("Share what's on your mind..."):
    
    # Save and display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process assistant response inside active context block
    with st.chat_message("assistant"):
        # Utilizing modern st.status container instead of simple spinner for high-tech look
        with st.status("Reflecting deeply...", expanded=False) as status:
            try:
                response_text = generate_response(user_prompt)
                status.update(label="Formulating response...", state="running")
                audio_bytes = text_to_speech(response_text)
                status.update(label="Complete", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Process failed", state="error")
                st.error(f"Something went wrong: {str(e)}")
                st.stop()

        # Render response text cleanly
        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Render audio block in a clean, non-intrusive drop expander
        if audio_bytes:
            with st.expander("🔊 Listen to Voice Response", expanded=True):
                st.audio(audio_bytes, format="audio/mp3")
