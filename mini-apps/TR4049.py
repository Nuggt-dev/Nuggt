import os
from apikey import apikey 
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import pandas as pd
import random

os.environ["OPENAI_API_KEY"] = apikey   

st.title("TR4049 Evidence Generator")
# prompt = st.text_input("Enter an age for customer segment")

age_template = PromptTemplate(
input_variables = ["age"],
template="""
Context: I am doing market validation for my Grocery Optimization app. We are speaking with 
a variety of individuals from different age groups, family sizes, careers etc. The user persona I am looking at is
"the family guy", who is a male/female with kids and usually does grocery shopping 1-2 times a week on the weekends. 
The interviews are conducted at Costco, Safeway, Walmart etc in the Bay Area, California.

Create a 1-liner profile for a user of the following age: {age}.

The 1-liner must fit the context. 

For example, if the age is 25, the description would be "Financial analyst who works in a VC firm. Dresses properly and oftentimes eats out or orders takeaway. 
Cooks only when he has family or friends around."

Another example for the age 50 is "A married father of two adult children. He works as a marketing manager and enjoys 
spending his weekends playing basketball and watching sports with his family."
 
One final example for age 35 is "A married mother of two young children. She works as a nurse and enjoys spending her 
weekends going to the park with her family and trying out new recipes in the kitchen."

Output: 
"""
)

description_template = PromptTemplate(
input_variables = ["description"],
template="""
Context: Here's the description of the user I am interviewing: {description}

The context is that I have the following hypotheses from hypothesis 13 to hypothesis 21. They are labelled as H13-H21 below.

H13: Family guys prioritise saving time over saving money
H14: Family guys prioritise saving time by shopping at a supermarket with a less crowded carpark.
H15: Family guys prioritise supermarkets that are nearer to their household.
H16: To save time, family guys are agnoistic to brands and look to purchase whichever groceries are most available and instock
H17: Family guys would prioritise shopping at supermarkets with less crowded checkouts.
H18: Family guys are influenced by deals, coupons, vouchers in their purchasing behavior.
H19: Family guys prefer bulk purchases over individual purchases.
H20: Family guys prioritize house brands over premium brands.
H21: Family guys prioritize good deals in-person over normal prices online.

For these hypotheses I want to generate this sort of response. For example,

If the description is, "A 46-year-old single mother of three teenagers, working as a high school teacher. 
Enjoys cooking healthy meals for her family and spends weekends attending her kids' sports events and exploring local farmers' markets."

"
General comments: Goes to Costco one a week with her family on the weekends and the local farmers market on the weekdays to buy organic food items.

H13: Validated. Money isn't a concern for her as long as the food she buys is healthy.
H14: Invalidated. She prefers less crowded areas.
H15: Validated. Prefers to go to Costco which is near her place.
H16: Invalidated. She cares about the brand she buys.
H17: Validated. She definitely prefers shorter queues and finds queueing the most time-consuming part of grocery shopping.
H18: Neutral. She would use a deal if the grocery item is something she really wants to buy.
H19: Validated. She prefers to buy in build because her kids more or less have the same eating habits.
H20: Validated. For basic groceries, house brands are preferred.
H21: Validated. She prefers to buy in person because going shopping as a fun activity she does with her kids.
"

So I want to generate this in the form,

General comments:

H13:
H14:
H15:
H16:
H17:
H18:
H19:
H20:
H21:
H22:
H23:
H24:

Each of the hypothesis responses must be printed on a new line.
"""
)

hypotheses_template = PromptTemplate(
input_variables = ["hypotheses"],
template="""
Context: Based on the following {hypotheses}, generate an Evaluation/Insight paragraph. Be slightly more
creative here and give a unique answer.

For example,
Interviewee grocery shops twice a week with her spouse, prefers fresh produce, and likes trying new recipes. 
She finds shopping time-consuming and uses a grocery list to ensure necessary items are purchased. 
While not keen on visiting multiple stores, she is willing to do so if necessary. 
The interviewee is open to paying more for quality and organic ingredients and shops twice a week to ensure the availability of fresh produce.

Output: 
"""
)

age_memory = ConversationBufferMemory(input_key="age", memory_key="chat_history")
description_memory = ConversationBufferMemory(input_key="description", memory_key="chat_history")
hypotheses_memory = ConversationBufferMemory(input_key="hypotheses", memory_key="chat_history")

llm=OpenAI(temperature=0.8, model_name="gpt-4")
age_chain = LLMChain(llm=llm, prompt=age_template, verbose=True, output_key="description", memory=age_memory)
description_chain = LLMChain(llm=llm, prompt=description_template, verbose=True, output_key="hypotheses", memory=description_memory)
hypotheses_chain = LLMChain(llm=llm, prompt=hypotheses_template, verbose=True, output_key="evaluation", memory=hypotheses_memory)

# if prompt:
results = []

for i in range(10):
    age = str(random.randint(25, 60))

    description = age_chain.run(age)
    hypotheses = description_chain.run(description=description)
    evaluation = hypotheses_chain.run(hypotheses=hypotheses)

    result = {
        "description": description,
        "hypotheses": hypotheses,
        "evaluation": evaluation,
    }

    # Append the dictionary to the results list
    results.append(result)
    st.write(result)

#Example response
# {
#   "description": "A 52-year-old software engineer and father of three teenage children, enjoys weekend family barbecues and grocery shopping at Costco to find deals for his household.",
#   "hypotheses": "General comments: This 52-year-old software engineer enjoys weekend family barbecues and finds deals at Costco for his household. He is a father of three teenage children and focuses on providing for his family.\n\nH13: Validated. As a busy father, he values time-saving options when shopping for groceries to spend more time with his family.\nH14: Validated. Prefers shopping at supermarkets with less crowded carparks to save time and avoid stress.\nH15: Validated. Prioritises supermarkets that are close to his household for convenience and time-saving purposes.\nH16: Validated. While brand preference might not be a significant factor, he focuses on the availability and stock of groceries to save time.\nH17: Validated. Prefers shopping at supermarkets with less crowded checkouts to reduce time spent waiting in line.\nH18: Validated. As a deal-seeker, he is influenced by deals, coupons, and vouchers when making purchasing decisions.\nH19: Validated. Prefers bulk purchases to save time and money by reducing the frequency of grocery shopping trips.\nH20: Validated. He prioritizes house brands over premium brands as a cost-effective option for his family.\nH21: Validated. Prefers finding good deals in-person over normal prices online, as it also provides an opportunity for a family outing.",
#   "evaluation": "This 52-year-old software engineer places a high value on time-saving and convenience when it comes to grocery shopping for his family. He prefers supermarkets with less crowded carparks and checkouts and prioritizes those near his home. Although brand preference is not a significant factor for him, the availability and stock of groceries are essential. As a deal-seeker, he is influenced by deals, coupons, and vouchers and tends to make bulk purchases to save both time and money. House brands are his go-to choice for cost-effective shopping, and he enjoys finding good deals in-person over shopping online. This not only helps him save money but also serves as an opportunity for a family outing, allowing him to spend quality time with his three teenage children."
# }