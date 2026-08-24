from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.chains import create_retrieval_chain

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding = embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-5.6-luna",
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the user's question using ONLY the provided context.

        If the answer cannot be found in the context,
        say: "I don't know based on the provided context."

        Context: {context}
        Question: {input}
        """
    )

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm = model,
        prompt = prompt,
    )

    retriever = vectorStore.as_retriever(search_kwargs = {"k": 1})

    retrieval_chain = create_retrieval_chain(
        retriever = retriever,
        combine_docs_chain = chain,
    )

    return retrieval_chain

docs = get_documents_from_web("https://www.aurelio.ai/learn/langchain-lcel")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is LCEL?",
})

print(response["answer"])

print(response["context"])







