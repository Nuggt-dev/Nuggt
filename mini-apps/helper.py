import os
from apikey import apikey 
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

def getAnswerBlock(title, inputVariable, promptTemplate, llm):
    prompt_template = PromptTemplate(
    input_variables = [inputVariable],
    template=promptTemplate
    )   
    chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, output_key=inputVariable + "_answer")
    answer = chain.run(inputVariable)
    st.subheader(title)
    st.write(answer)