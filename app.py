import os
import re
import yt_dlp
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="centered"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
    }
    .main { background-color: #0f0f0f; color: #f0f0f0; }
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        border: 1px solid #333;
        color: #f0f0f0;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff0000, #cc0000);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(255, 0, 0, 0.4);
    }
    .info-box {
        background: #1a1a1a;
        border-left: 3px solid #ff0000;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #aaa;
    }
    .success-box {
        background: #0d1f0d;
        border-left: 3px solid #00cc44;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #aaffaa;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# API Key Setup
# ─────────────────────────────────────────────
def get_api_key():
    return os.getenv("OPENAI_API_KEY", "")

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def extract_video_id(url: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url.strip()):
        return url.strip()
    return None


def parse_vtt(vtt_path: str) -> str:
    """Extract plain text from a .vtt subtitle file."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r'WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
    content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> .*?\n', '', content)
    content = re.sub(r'<[^>]+>', '', content)   # remove HTML tags like <c>
    content = re.sub(r'\n+', ' ', content)       # replace newlines with space
    content = re.sub(r'\s+', ' ', content)       # remove extra spaces

    return content.strip()


def get_transcript(video_id: str) -> str:
    """Fetch transcript using yt-dlp (avoids YouTube IP blocking)."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        'writeautomaticsub': True,
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'skip_download': True,          # don't download the video
        'outtmpl': f'/tmp/{video_id}',
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    # yt-dlp saves subtitles as .vtt file
    vtt_path = f"/tmp/{video_id}.en.vtt"

    if not os.path.exists(vtt_path):
        raise Exception("No English transcript/subtitles found for this video.")

    transcript_text = parse_vtt(vtt_path)

    # Clean up temp file
    os.remove(vtt_path)

    return transcript_text


def build_chain(transcript: str, api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embed + store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Prompt
    prompt = PromptTemplate(
        template="""
You are a helpful assistant that answers questions about a YouTube video.
Answer ONLY from the provided transcript context.
If the context is insufficient or the answer is not in the transcript, clearly say:
"I don't have enough information from the video to answer that."

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "video_url" not in st.session_state:
    st.session_state.video_url = ""

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🎬 YouTube RAG Chatbot")
st.markdown("*Ask anything about a YouTube video — powered by transcripts + GPT-4o mini*")
st.divider()

# Load API key from .env automatically
api_key = get_api_key()

# Video URL input
url_input = st.text_input(
    "📺 YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=QoKpQMJnBHY",
    value=st.session_state.video_url
)

col1, col2 = st.columns([1, 3])
with col1:
    load_btn = st.button("Load Video", use_container_width=True)
with col2:
    if st.session_state.video_loaded:
        st.markdown('<div class="success-box">✅ Video loaded and ready!</div>', unsafe_allow_html=True)

# Load video logic
if load_btn:
    if not api_key:
        st.error("OpenAI API key not found. Please check your .env file.")
    elif not url_input.strip():
        st.error("Please enter a YouTube video URL.")
    else:
        video_id = extract_video_id(url_input.strip())
        if not video_id:
            st.error("Could not extract a valid YouTube video ID from the URL.")
        else:
            with st.spinner("Fetching transcript and building knowledge base..."):
                try:
                    transcript = get_transcript(video_id)
                    chain = build_chain(transcript, api_key)
                    st.session_state.chain = chain
                    st.session_state.chat_history = []
                    st.session_state.video_loaded = True
                    st.session_state.video_url = url_input.strip()
                    st.success("Video loaded! Start asking questions below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────
if st.session_state.chain:
    st.divider()
    st.markdown("### 💬 Chat")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if question := st.chat_input("Ask something about the video..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.chain.invoke(question)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.markdown("""
<div class="info-box">
👆 Enter a YouTube URL above and click <strong>Load Video</strong> to get started.<br><br>
The app will fetch the transcript and let you ask any question about the video content.
If the answer isn't in the transcript, it will tell you so.
</div>
""", unsafe_allow_html=True)





