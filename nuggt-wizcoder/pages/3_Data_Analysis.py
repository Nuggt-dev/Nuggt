from nuggt import *
from helper.sidebar_functions import sidebar_logo

sidebar_logo("assets/nuggt-logo.png")

user_input = """Print the head of {upload:data} with print() using {tool:python}. Create a bar plot between {text:column1} column and {text:column2} column with matplotlib using {tool:python}. Display the bar plot with proper labels with streamlit library using {tool:python}"""
output_format = "I have displayed the visualisations using streamlit"

st.subheader("Data Analysis")
st.markdown("""Upload an Excel/CSV file for data analysis. You can ask the app generic questions like "summarise this data" or specific questions like "What is the relationship between Column A and Column B". The app can also generate data visualisations like bargraphs, piecharts etc. You can generate visualisations by typing "Generate visualisations for..". This app was created with a single AI command,             
""")
st.markdown("""Open **:red[{ upload : file }]** using **:green[{ tool : python }]**. Display its description using **:green[{ tool : python }]** using **:green[{ tool : python }]**. Complete the task **:blue[{ text : input }]** using **:green[{ tool : python }]**. Display your results using **:green[{ tool : data_analysis }]**.  \n
""")

if user_input and output_format:
    variables = extract_variables(user_input)
    nuggt(user_input, output_format, variables)

    














