import os
from apikey import apikey 
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ["OPENAI_API_KEY"] = apikey   

st.title("🦄 STARTUP GPT 🔥")
prompt = st.text_input("Enter customer profile")

customer_template = PromptTemplate(
input_variables = ["customer"],
template="give me a problem statement faced by {customer}"
)

problem_template = PromptTemplate(
input_variables = ["problem_statement", "wikipedia"],
template="give me a startup ideas to solve the problem: {problem_statement} by leveraging this wikipedia research: {wikipedia}"
)

idea_template = PromptTemplate(
input_variables = ["idea"],
template="Give me a company name for the idea: {idea}"
)

email_template = PromptTemplate(
input_variables = ["name"],
template="Give me a marketing email to pitch this company: {name}"
)

customer_memory = ConversationBufferMemory(input_key="customer", memory_key="chat_history")
problem_memory = ConversationBufferMemory(input_key="problem_statement", memory_key="chat_history")
idea_memory = ConversationBufferMemory(input_key="idea", memory_key="chat_history")
name_memory = ConversationBufferMemory(input_key="name", memory_key="chat_history")

llm=OpenAI(temperature=1, model_name="gpt-4")
customer_chain = LLMChain(llm=llm, prompt=customer_template, verbose=True, output_key="problem_statement", memory=customer_memory)
problem_chain = LLMChain(llm=llm, prompt=problem_template, verbose=True, output_key="idea", memory=problem_memory)
idea_chain = LLMChain(llm=llm, prompt=idea_template, verbose=True, output_key="name", memory=idea_memory)
email_chain = LLMChain(llm=llm, prompt=email_template, verbose=True, output_key="email", memory=name_memory)

wiki = WikipediaAPIWrapper()
# sequential_chain = SequentialChain(chains=[customer_chain, problem_chain, idea_chain, email_chain], input_variables=["customer"], output_variables=["problem_statement", "idea", "name", "email"], verbose=True)

if prompt:
    problem = customer_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    idea = problem_chain.run(problem_statement=problem, wikipedia=wiki_research)
    name = idea_chain.run(idea=idea)
    email = email_chain.run(name=name)


    st.subheader("😝 Problem Statement")
    st.write(problem)
    st.subheader("💡 Idea")
    st.write(idea)
    st.subheader("🫦 Name")
    st.write(name)
    st.subheader("📧 Email Template")
    st.write(email)

    with st.expander("Customer History"):
        st.info(customer_memory.buffer)

    with st.expander("Problem History"):
        st.info(problem_memory.buffer)

    with st.expander("Wiki History"):
        st.info(wiki_research)

    with st.expander("Idea History"):
        st.info(idea_memory.buffer)

    with st.expander("Name History"):
        st.info(name_memory.buffer)


# if prompt:
#     response = sequential_chain({"customer": prompt})
#     st.subheader("😝 Problem Statement")
#     st.write(response["problem_statement"])
#     st.subheader("💡 Idea")
#     st.write(response["idea"])
#     st.subheader("🫦 Name")
#     st.write(response["name"])
#     st.subheader("📧 Email Template")
#     st.write(response["email"])

#     with st.expander("🧠 Memory"):
#         st.info(memory.buffer)
