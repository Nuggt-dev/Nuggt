from nuggt import *
from helper.sidebar_functions import sidebar_logo

sidebar_logo("assets/nuggt-logo.png")

user_input = """Print the head of {upload:data} with print() using {tool:python}. Create a bar plot between {text:column1} column and {text:column2} column with matplotlib using {tool:python}. Display the bar plot with proper labels with streamlit library using {tool:python}"""
output_format = "I have displayed the visualisations using streamlit"

st.subheader("Data Analysis")
st.markdown("""The following prompt was used to create this Nuggt App:""")
st.markdown("""Print the head of {upload:data} with print() using {tool:python}. Create a bar plot between {text:column1} column and {text:column2} column with matplotlib using {tool:python}. Display the bar plot with proper labels with streamlit library using {tool:python}\n""")

if user_input and output_format:
    variables = extract_variables(user_input)
    nuggt(user_input, output_format, variables)

    














