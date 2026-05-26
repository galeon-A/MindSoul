import streamlit as st
import tempfile
from gtts import gTTS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    PubMedAPIWrapper,
)

from langchain.docstore.document import Document


# -----------------------------------
# Streamlit Config
# -----------------------------------
st.set_page_config(
    page_title="MindSoul AI Therapist",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.title("⚙️ Settings")

google_api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)

st.sidebar.markdown("""
### About
MindSoul is an AI-powered therapist chatbot using:

- Gemini AI
- LangChain
- FAISS
- Wikipedia
- PubMed

⚠️ This is not a replacement for professional therapy.
""")

# -----------------------------------
# Title
# -----------------------------------
st.title("🧠 MindSoul AI Therapist")
st.caption("Your AI-powered mental wellness companion")

# -----------------------------------
# Stop if no API key
# -----------------------------------
if not google_api_key:
    st.warning("Please enter your Gemini API key in the sidebar.")
    st.stop()

# -----------------------------------
# LLM
# -----------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key,
    temperature=0.7,
)

# -----------------------------------
# External Sources
# -----------------------------------
wiki = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=500
)

pubmed = PubMedAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=500
)

# -----------------------------------
# Vector Store Function
# -----------------------------------
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
        google_api_key=google_api_key
    )

    vector_store = FAISS.from_documents(
        docs,
        embeddings
    )

    return vector_store


# -----------------------------------
# Retrieve Context
# -----------------------------------
def retrieve_context(query):

    try:
        wiki_data = wiki.run(query)
    except:
        wiki_data = ""

    try:
        pubmed_data = pubmed.run(query)
    except:
        pubmed_data = ""

    combined_text = f"{wiki_data}\n\n{pubmed_data}"

    if not combined_text.strip():
        return "No additional context found."

    vector_store = create_vector_store(combined_text)

    relevant_docs = vector_store.similarity_search(
        query,
        k=3
    )

    return "\n".join(
        [doc.page_content for doc in relevant_docs]
    )


# -----------------------------------
# Generate Response
# -----------------------------------
def generate_response(user_query):

    context = retrieve_context(user_query)

    prompt = f"""
You are a compassionate AI therapist.

Guidelines:
- Be empathetic and emotionally supportive
- Suggest healthy coping strategies
- Never diagnose severe mental illnesses
- Encourage professional help when necessary
- Keep responses warm and conversational

Context:
{context}

User:
{user_query}

Therapist:
"""

    response = llm.invoke(prompt)

    return response.content


# -----------------------------------
# Text to Speech
# -----------------------------------
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


# -----------------------------------
# Chat History
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------
# Display Messages
# -----------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------------
# User Input
# -----------------------------------
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

        with st.spinner("Thinking..."):

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
