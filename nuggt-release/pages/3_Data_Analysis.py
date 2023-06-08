from Nuggt_Playground import *
from helper.sidebar_functions import sidebar_logo

st.set_page_config(page_title="Data Analysis", layout="wide")

sidebar_logo("assets/nuggt-logo.png")

user_input = """Step: Open {upload:file} using {tool:python}
Step: Display its description using {tool:python}
Step: Complete the task {text:input} using {tool:python}
Step: Display your results using {tool:display}
"""
output_format = "I have displayed the visualisations using streamlit"

st.subheader("Data Analysis")
st.markdown("""Upload an Excel/CSV file for data analysis. You can ask the app generic questions like "summarise this data" or specific questions like "What is the relationship between Column A and Column B". The app can also generate data visualisations like bargraphs, piecharts etc. You can generate visualisations by typing "Generate visualisations for..". This app was created with a single AI command,             
""")
st.markdown("""Open **:red[{ upload : file }]** using **:green[{ tool : python }]**. Display its description using **:green[{ tool : python }]** using **:green[{ tool : python }]**. Complete the task **:blue[{ text : input }]** using **:green[{ tool : python }]**. Display your results using **:green[{ tool : data_analysis }]**.  \n
""")

if user_input and output_format:
    save_to_sheets(user_input, output_format, "-", "-")
    variables = extract_variables(user_input)
    nuggt(user_input, output_format, variables)

    














