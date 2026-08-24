from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# Create Retriever
loader = WebBaseLoader("https://www.aurelio.ai/learn/langchain-lcel")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(splitDocs, embedding=embedding)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    use_responses_api=True,
)
search = TavilySearch(
    max_results=1,
    topic="general",
)

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name = "lcel_search", # the name of this tool
    description="Use this tool when searching for information about Langchain Expression Language (LCEL)"
)

tools = [search, retriever_tool]

agent = create_agent(
    model=llm,
    system_prompt=
    """
    You are a friendly assistant called Max.

    For every question about LangChain Expression Language (LCEL),
    you MUST use the lcel_search tool before answering.

    For questions requiring current information, use the Tavily search tool.
    """,
    tools=tools,
)

def process_chat(agent, user_input, chat_history):
    messages = chat_history + [
        HumanMessage(content = user_input)
    ]

    response = agent.invoke({
        "messages": messages
    })

    print("\n--- AGENT TRACE ---")
    for message in response["messages"]:
        message.pretty_print()
    print("--- END TRACE ---\n")

    return response

# for message in response["messages"]:
#   message.pretty_print()

if __name__ == "__main__":
    chat_history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = process_chat(
            agent,
            user_input,
            chat_history
        )

        ai_message = response["messages"][-1] #-1 От всички messages, които agent-ът е генерирал по време на изпълнението, дай ми последното съобщение.

        chat_history.append(
            HumanMessage(content=user_input)
        )

        chat_history.append(ai_message)

        print("Assistant:", ai_message.content)




