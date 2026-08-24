from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Instantiate Model
llm = ChatOpenAI(
    temperature = 0.7,
    model = "gpt-3.5-turbo-1106",
)

# Prompt Template
 # First way:
    # prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}")
# Second way:
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word. Return the result as a comma separated list."),
        ("human", "{input}")
    ]
)

# create LLM Chain
chain = prompt | llm

response = chain.invoke({"input": "happy"})
print(type(response))