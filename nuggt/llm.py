from langchain.agents import load_tools, initialize_agent, AgentType, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import OpenAI, LLMChain
from langchain.tools import BaseTool
from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.utilities import GoogleSearchAPIWrapper, SerpAPIWrapper, GoogleSerperAPIWrapper
from typing import List, Union
import re
import os

#os.environ["GOOGLE_CSE_ID"] = "a60ee944812a441d9"
#os.environ["GOOGLE_API_KEY"] = "AIzaSyBq3rhqM03-hqLbDqHeKHbc8K2qSgqMW7Q"
os.environ["SERPER_API_KEY"] = "9cae0f9d724d3cb2e51211d8e49dfbdc22ab279b"

search = GoogleSerperAPIWrapper()
#search = SerpAPIWrapper()

llm = OpenAI(temperature=0, openai_api_key="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")

#Tools

def check(input):
    check_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Check if {question} is a valid arithmetic question in Addition, Substraction, Division or Multiplication. Answer in 'valid' or 'invalid'. Do not output anything else but 'Yes' or 'No'
    Examples:
    What is 10+20?: valid
    what is 10]+(20)?: invalid
    calculate (10) + (20): valid
    10*10/20*20: valid
    (10*10))/(20*20): invalid
    ((10*10))/(20*20): valid
    """,
    )
    chain = LLMChain(llm=llm, prompt=check_prompt)
    #return chain.run(input).strip()
    return "valid"

def invalid(input):
    return "tool reported"

def happy(_):
    return "happy"
tools = [
    Tool(
        name = "check",
        func=check,
        description="This is always the first tool you use to check if the question is valid."
    ),
    Tool(
        name = "report invalid tool",
        func=invalid,
        description="useful for when you do not have a valid tool."
    ),
    Tool(
        name = "Google Search",
        description="useful for getting information.",
        func=happy
    )
]


# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
agent_executor.run("What is the color of the sky?")

"""
Following are some examples:

Example 2:
Question: What is the circumference of earth?
Thought: I should check if the given input is valid.
Action: check
Action Input: What is the circumference of earth?
Observation: invalid
Thought: I now have the final answer. 
Final Answer: Please input a valid question

Example 3:
Question: What is the color of the sky?
Thought: I should check if the given input is valid.
Action: check
Action Input: What is the color of the sky?
Observation: valid
Thought: I should use the search tool to find the answer. 
Action: search
Action Input: What is the color of the sky?
Observation:search is not a valid tool, try another one.
Thought: I should report that I do not have a valid tool
Action: report invalid tool
Action Input: search is not a valid tool
Observation: tool reported
Thought: I now have the final answer. 
Final Answer: I do not have a valid tool to answer this question.

"""

"""
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Following is an example:

Example 2:
Question: What is the circumference of earth?
Thought: I should check if the given input is valid.
Action: check
Action Input: What is the circumference of earth?
Observation: invalid
Thought: I now have the final answer. 

"""


