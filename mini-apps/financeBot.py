import os
from apikey import apikey 
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from helper import getAnswerBlock

os.environ["OPENAI_API_KEY"] = apikey   

st.set_page_config(page_title="Finance Bot", page_icon=":moneybag:")
st.title("Finance Bot")

form = st.form("my_form")

tickerSymbol = form.text_input("Enter the ticker symbol you would like to learn about:")

options = form.multiselect(
'What would you learn to learn about?',
['Financial Projections', 'Revenue Model', 'Future Outlook', "Competitors", "Market Size", "Market Share", "Market Growth", "Market Trends", "Market Segmentation", "Market Analysis", "Market Research", "Market Opportunity", "Market Potential", "Market Demand", "Market Saturation", "Market Entry", "Market Positioning", "Market Strategy", "Market Penetration", "Market Development", "Market Expansion", "Market Diversification", "Market Risk", "Market Assessment"],
['Financial Projections', 'Revenue Model', 'Future Outlook', "Competitors"])

submitted = form.form_submit_button("Submit")

llm=OpenAI(temperature=0, model_name="gpt-4")

if submitted:
    if "Financial Projections" in options:
        getAnswerBlock(
            title = "Financial Projections",
            inputVariable = "ticker",
            promptTemplate = "Give me the financial projects for {ticker}",
            llm = llm
        )

    if "Revenue Model" in options:
        getAnswerBlock(
            title = "Revenue Model",
            inputVariable = "ticker",
            promptTemplate = "Give me the revenue statement for {ticker}",
            llm = llm
        )

