from Nuggt_Playground import *
from sidebar_functions import sidebar_logo

st.set_page_config(page_title="Search", layout="wide")

sidebar_logo("nuggt-logo.png")

promotion = """Nuggt allows you to create deployable AI apps with a single command. 
You can create an app for anything you imagine. For example, the following command would create an app that researches
on a given topic and creates a powerpoint presentation with the relevant sources:
`Gather information on {text:topic} using {tool:google} and based on the gathered information create a powerpoint presentation with the relevant sources using {tool:powerpoint}` 
"""
user_input = """Step: Find websites related to {text:query} using {tool:google}.
Step: Browse the results to gather information on {text:query} using {tool:browse_website}.
Step: Based on the gathered information answer {text:query}"""

output_format = "For each source output the following:\nContent: The relevant information you found in that website\nSource: The link of that website"


st.subheader("Search")
st.markdown("Ask a query and Nuggt would provide you answers along with the relevant sources.")
st.markdown("""This app was created with a single AI command - 
            **Find websites related to :blue[{text : query}] using :green[{tool : google}] and list the relevant sources.**  \n
""")
st.markdown("""**- :blue[query]** is a variable that defines your search query.
""")
st.markdown("""**- :green[google]** is a tool used for search.
""")
        
if user_input and output_format:
    save_to_sheets(user_input, output_format, "-", "-")
    variables = extract_variables(user_input)
    nuggt(user_input, output_format, variables)














