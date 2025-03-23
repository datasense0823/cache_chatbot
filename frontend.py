import os
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

# -------------------- Setup --------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "semantic-cache"

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def embed_user_question(question: str) -> list:
    return embedding_model.embed_query(question)

def llm_call(query: str):
    llm = ChatGroq(temperature=0, model="llama3-70b-8192")
    template = """You are an AI Tool that Can Answer Anything.

    Given the User Question, Answer to the Best of your knowledge in 50 words

    Question: {question}
    """
    prompt = PromptTemplate(input_variables=["question"], template=template)
    chain = prompt | llm
    result = chain.invoke({"question": query})
    return result.content.strip()

def get_answer(user_question: str, similarity_threshold: float):
    query_embedding = embed_user_question(user_question)
    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)

    if results and results['matches']:
        top_match = results['matches'][0]
        score = top_match['score']
        if score >= similarity_threshold:
            return {
                "answer": top_match["metadata"]["answer"],
                "source": "cache",
                "matched_question": top_match["metadata"]["question"],
                "similarity": round(score, 4),
                "timestamp": top_match["metadata"].get("timestamp")
            }

    answer = llm_call(user_question)
    index.upsert([
        {
            "id": str(uuid.uuid4()),
            "values": query_embedding,
            "metadata": {
                "question": user_question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    ])
    return {
        "answer": answer,
        "source": "llm",
        "matched_question": None,
        "similarity": 0.0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Semantic LLM Playground", layout="centered")
st.title("🧠 Semantic Cache Q&A Playground")
st.markdown("Explore intelligent caching with OpenAI Embeddings + Pinecone + Groq")

# Sidebar options
with st.sidebar:
    st.header("🔧 Playground Settings")
    similarity_threshold = st.slider("Similarity Threshold", 0.5, 0.95, 0.7, 0.01)
    show_details = st.checkbox("Show technical details", value=True)
    st.markdown("You can modify the threshold to experiment with cache sensitivity.")
    st.markdown("Coming soon: history export, multiple match results, custom prompts.")

# Input box
user_input = st.text_input("💬 Enter your question")
if st.button("🚀 Get Answer") and user_input:
    with st.spinner("Crunching vectors & fetching insights..."):
        wall_start = time.time()
        result = get_answer(user_input, similarity_threshold)
        wall_end = time.time()

    st.success("✅ Here's your answer!")

    st.markdown(f"""
    <div style='padding: 10px; border-left: 5px solid #4CAF50; background-color: #f9f9f9;'>
        <strong>🧠 Answer:</strong> {result['answer']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    **📌 Source:** `{result['source']}`  
    **🕒 Timestamp:** `{result['timestamp']}`  
    """)

    if result['matched_question']:
        st.markdown("""
        **🔁 Matched Question:** {}  
        **🔍 Similarity Score:** {}  
        """.format(result['matched_question'], result['similarity']))

    if show_details:
        wall_time = (wall_end - wall_start) * 1000
        st.code(f"⏱️ Wall time: {wall_time:.1f} ms")
        st.caption("Performance metrics will improve on cache hits.")
