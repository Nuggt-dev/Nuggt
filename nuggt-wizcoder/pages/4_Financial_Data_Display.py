from nuggt import *
from helper.sidebar_functions import sidebar_logo

sidebar_logo("assets/nuggt-logo.png")

st.subheader("Financial Data Display")

user_input = """Get past 7 days OHLC data for stock {text:stock} using {tool:python}. Calculate the MACD and save the MACD table as {text:file} using {tool:python}. Save line plots between MACD and Signal data as {text:image} using {tool:python}."""

output_format = "I have saved the tables and visualisations for the given stock."

st.markdown("""Query any tabular financial data. This app was created with a single AI command,
""")

st.markdown("""
 Get past 7 days OHLC data for stock **:blue[{ text : stock }]** using **:green[{ tool : python }]**. Calculate the MACD using **:green[{ tool : python }]** (Do not use talib). Display the results in a table using **:green[{ tool : display }]**. Display basic visualisations using **:green[{ tool : display }]**.
""")

if user_input and output_format:
    variables = extract_variables(user_input)
    nuggt(user_input, output_format, variables)












