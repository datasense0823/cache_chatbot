import os
import time
import resource
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# -------------------- Setup --------------------
load_dotenv()

# API Keys from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Init Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "semantic-cache"

# Create Pinecone index if not exists
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # embedding dim of text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to Pinecone index
index = pc.Index(index_name)

# -------------------- OpenAI Embeddings via LangChain --------------------
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def embed_user_question(question: str) -> list:
    return embedding_model.embed_query(question)

# -------------------- Groq LLM Response via LangChain --------------------
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

# -------------------- Semantic Caching Logic --------------------
def get_answer(user_question: str):
    query_embedding = embed_user_question(user_question)

    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)

    if results and results['matches']:
        top_match = results['matches'][0]
        score = top_match['score']
        if score >= 0.7:
            return {
                "answer": top_match["metadata"]["answer"],
                "source": "cache",
                "matched_question": top_match["metadata"]["question"],
                "similarity": round(score, 4)
            }

    # If no match, call LLM
    answer = llm_call(user_question)

    # Store question+answer in Pinecone
    index.upsert([
        {
            "id": str(uuid.uuid4()),
            "values": query_embedding,
            "metadata": {
                "question": user_question,
                "answer": answer
            }
        }
    ])

    return {
        "answer": answer,
        "source": "llm",
        "matched_question": None,
        "similarity": 0.0
    }

# -------------------- CLI Loop with Timer --------------------
if __name__ == "__main__":
    print("🤖 Welcome to Semantic Cache (OpenAI + Pinecone + Groq)")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        wall_start = time.time()
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)

        result = get_answer(question)

        wall_end = time.time()
        cpu_end = resource.getrusage(resource.RUSAGE_SELF)

        user_time = cpu_end.ru_utime - cpu_start.ru_utime
        sys_time = cpu_end.ru_stime - cpu_start.ru_stime
        total_cpu = user_time + sys_time
        wall_time = wall_end - wall_start

        print(f"\n📌 Source: {result['source']}")
        if result['matched_question']:
            print(f"🔁 Matched Question: {result['matched_question']}")
            print(f"🔍 Similarity: {result['similarity']}")

        print(f"\n🧠 Answer: {result['answer']}")

        print(f"\n⏱️ Performance:")
        print(f"CPU times: user {user_time * 1000:.1f} ms, sys: {sys_time * 1000:.1f} ms, total: {total_cpu * 1000:.1f} ms")
        print(f"Wall time: {wall_time * 1000:.1f} ms\n")
