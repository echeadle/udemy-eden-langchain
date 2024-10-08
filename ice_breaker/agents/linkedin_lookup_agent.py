import os
import sys 
from dotenv import load_dotenv
load_dotenv()
pythonpath = os.getenv("PYPATH")
print(f'Python path 1={pythonpath}')
if pythonpath and pythonpath not in sys.path:
    sys.path.append(pythonpath)
print(sys.path)
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
from tools.tools import get_profile_url_tavily


def lookup(name: str)->str:
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini", 
    )
    template = """given the full name {name_of_person} I want you to get me a link to their Linkedin profile page.
                            Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need to get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools_for_agent, verbose=True)
    
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )
    
    linked_profile_url = result["output"]
    return linked_profile_url

if __name__== "__main__":
    pythonpath = os.getenv("PYPATH")
    print(f'Python path 1={pythonpath}')
    if pythonpath and pythonpath not in sys.path:
        sys.path.append(pythonpath)
    print(sys.path)

    linkedin_url = lookup(name="Edward Cheadle")
    print(linkedin_url)