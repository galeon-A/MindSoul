# 🧠 MindMate AI Therapist

MindMate AI Therapist is an AI-powered mental wellness assistant built using:

- Streamlit
- Google Gemini
- LangChain
- FAISS
- Wikipedia
- PubMed

## Features

- AI Therapist Chat
- Context-Aware Responses
- PubMed + Wikipedia Knowledge
- Voice Responses (Text-to-Speech)
- Streamlit UI
- Vector Search using FAISS

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/mindmate-ai-therapist.git
cd mindmate-ai-therapist
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GOOGLE_GEMINI_API_KEY=your_api_key_here
```

Get API Key from:
https://aistudio.google.com/app/apikey

---

## Run Application

```bash
streamlit run app.py
```

---

## Deploy on Streamlit

1. Upload project to GitHub
2. Go to https://streamlit.io/cloud
3. Connect GitHub Repository
4. Deploy

---

## Tech Stack

- Python
- Streamlit
- LangChain
- Gemini AI
- FAISS
- PubMed API
- Wikipedia API

---

## Disclaimer

This project is for educational purposes only and is not a substitute for professional mental health treatment.