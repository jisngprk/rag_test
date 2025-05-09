# imports
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# load in the .env variables
load_dotenv()

print(os.environ["OPENAI_API_KEY"])
with open("data.txt", "r") as file:
    text = file.read()


text_splitter = CharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=50,
    length_function=len)

texts = text_splitter.split_text(text)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings
)
documents = [Document(page_content=text) for text in texts]
ids = vector_store.add_documents(documents)

query = "Explain about COVID effect in U.S"
results = vector_store.similarity_search(
    query,
    k=2
)
print(f"docs: {results}")

for result in results:
    print(f'* {result.page_content} [{result.metadata}] \n\n')


retriever = vector_store.as_retriever()
llm = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_template = """Use the context provided to answer the user's
question below. If you don't know the answer based on the context provided,
tell the user that you don't know the answer to thier question
based on the context provided and that you are sorry

context: {context}

question: {query}

answer: 
"""

custom_rag_prompt = PromptTemplate.from_template(prompt_template)

rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

ret = rag_chain.invoke(query)
print('--')
print(ret)
