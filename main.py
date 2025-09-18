import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch
from langchain import hub

load_dotenv()

os.getenv("")

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def main():
    result = chain.invoke(
        input={
            "input":"search for 3 ai engineering jobs using langchain in Singapore on LinkedIn and list their details."
        }
    )
    print(result)

if __name__ == "__main__":
    main()
