from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
load_dotenv()

text="""You are an AI Model. Reply to user as best as you can

User Question: {question}"""

prompt=PromptTemplate(
    input_variables=["question"],
    template=text
)

llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()])

chain=prompt|llm
response=chain.invoke({"question":"Tell me about India"})
print(response.content)


