import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# --- Config ---
load_dotenv()  # loads from .env file in project root
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-3.3-70b-instruct"
PDF_PATH = "./sample.pdf"

if not OPENROUTER_API_KEY:
    st.error("⚠️ Missing OpenRouter API key. Add OPENROUTER_API_KEY=your_key to your `.env` file.")
    st.stop()

# FIX 2: Guard against missing PDF before it causes a cryptic crash
if not os.path.exists(PDF_PATH):
    st.error(f"⚠️ PDF not found at `{PDF_PATH}`. Place your document there and restart.")
    st.stop()

# FIX 3: Correct refine chain prompts
# The 'refine' chain needs TWO prompts with the correct variable names.

# Initial prompt — used on the first retrieved chunk
initial_prompt_template = """You are a professional company assistant named Cela.
Rules:
- Only answer using the provided context
- If the answer is not in the context, say: "I don't have that information."
- Be concise and clear
- Do not hallucinate

Context:
{context_str}

Question: {question}

Answer:"""

initial_prompt = PromptTemplate(
    template=initial_prompt_template,
    input_variables=["context_str", "question"]
)

# Refine prompt — used on subsequent chunks to improve the answer
refine_prompt_template = """You are a professional company assistant named Cela.
You have an existing answer and new context to refine it.
Rules:
- Only use information from the provided context
- If the new context isn't useful, return the existing answer unchanged
- Be concise and clear
- Do not hallucinate

Existing Answer:
{existing_answer}

New Context:
{context_str}

Question: {question}

Refined Answer:"""

refine_prompt = PromptTemplate(
    template=refine_prompt_template,
    input_variables=["existing_answer", "context_str", "question"]
)

# FIX 4: Use `base_url` not `openai_api_base` (deprecated in newer langchain-openai)
llm = ChatOpenAI(
    model=MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.5,
    max_tokens=512,
    default_headers={
        "X-Title": "Cela",
    }
)

# --- PDF Index (cached per session) ---
@st.cache_resource
def load_index():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_index()

# --- RAG Chain ---
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    input_key="question",
    return_source_documents=False,
    chain_type_kwargs={
        "question_prompt": initial_prompt,   # FIX 3a: initial prompt required
        "refine_prompt": refine_prompt        # FIX 3b: correct key name
    }
)

# --- UI ---
st.title("Ask Cela")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

prompt = st.chat_input("Ask something about Celcom's Africa")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # FIX 5: Wrap in try/except so API errors don't crash the app
        try:
            result = chain.invoke({"question": prompt})
            response = result["result"]
        except Exception as e:
            response = f"⚠️ Something went wrong: {str(e)}"

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})