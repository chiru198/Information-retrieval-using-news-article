import os
import time
import nest_asyncio
import streamlit as st
import os
import streamlit as st

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


nest_asyncio.apply()


try:
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
        HarmBlockThreshold,
        HarmCategory
    )
    from langchain_community.document_loaders import UnstructuredURLLoader
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    st.error(f"❌ Missing package: {e}. Please install dependencies.")
    st.stop()

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="News Research Tool 📈",
    page_icon="📰",
    layout="wide"
)


st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff, #d9e4f5);
        color: #222;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1E3A8A;
        color: white;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] label {
        color: #fff;
    }
    
    /* Titles */
    h1 {
        color: #1E40AF;
        font-weight: 800;
    }
    h2, h3 {
        color: #2563EB;
    }

    /* Input box */
    .stTextInput>div>div>input {
        border: 2px solid #2563EB;
        border-radius: 8px;
        padding: 6px;
    }

    /* Buttons */
    button {
        background: #2563EB !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
    }
    button:hover {
        background: #1D4ED8 !important;
    }

    /* Answer box */
    .stSuccess {
        background: #DCFCE7;
        color: #166534;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #22C55E;
    }

    /* Sources */
    .stInfo {
        background: #DBEAFE;
        color: #1E3A8A;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📰 News Article URLs")
st.sidebar.markdown("Paste up to 3 URLs to fetch news content.")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("⚡ Process URLs")

st.title("📊 News Research Tool (with Google Gemini)")

with st.expander("ℹ️ About this app", expanded=False):
    st.markdown("""
    This tool lets you:
    - 🔗 Fetch news articles from URLs  
    - ✂️ Split text into chunks for better retrieval  
    - 🤖 Use **Google Gemini** for embeddings + Q&A  
    - 📂 Store data in **FAISS vector DB**  
    """)

faiss_path = "faiss_store_gemini"
main_placeholder = st.empty()

# -------------------------
# LLM Init
# -------------------------
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.6,
        max_output_tokens=500,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
except Exception as e:
    st.error(f"❌ Could not initialize Gemini. Check your GOOGLE_API_KEY. Error: {e}")
    llm = None


if process_url_clicked and llm is not None:
    with st.spinner("🔄 Fetching and processing data..."):
        loader = UnstructuredURLLoader(urls=[u for u in urls if u.strip() != ""])
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)

        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore_gemini = FAISS.from_documents(docs, embeddings)
            vectorstore_gemini.save_local(faiss_path)
            st.success("✅ FAISS index built and saved successfully!")
        except Exception as e:
            st.error(f"❌ Error creating embeddings: {e}")

# -------------------------
# Ask a Question
# -------------------------
st.markdown("---")
st.subheader("💬 Ask Questions from the Processed Articles")

query = st.text_input("🔎 Enter your question:")
if query:
    if llm is None:
        st.warning("⚠️ LLM not initialized. Please set your GOOGLE_API_KEY.")
    elif os.path.exists(faiss_path):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

        with st.spinner("🤖 Thinking..."):
            try:
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                st.success(f"### ✅ Answer\n\n{result['answer']}")

                sources = result.get("sources", "")
                if sources:
                    st.info("### 📌 Sources")
                    for src in sources.split("\n"):
                        st.markdown(f"- {src}")
            except Exception as e:
                st.error(f"❌ Retrieval error: {e}")
    else:
        st.error("⚠️ FAISS index not found. Please process URLs first.")
