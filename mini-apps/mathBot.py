import os
import ssl
from apikey import apikey 
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import WikipediaAPIWrapper

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["OPENAI_API_KEY"] = apikey   
os.environ["WOLFRAM_ALPHA_APPID"] = "A9H8XV-V6W57YW9KQ"

st.title("Math Bot")
prompt = st.text_input("Enter math problem:")

llm=OpenAI(temperature=0, model_name="gpt-4")

tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

if prompt:
    answer = agent.run(prompt)
    st.subheader("Answer:")
    st.write(answer)


