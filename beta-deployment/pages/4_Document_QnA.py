from Nuggt_Playground import *

st.set_page_config(page_title="Document QnA", layout="wide")
    
information = """This app was created with a single AI command as follows:
```Open {upload:file} using {tool:document_loader} and answer {text:query} with sources using {tool:document_chat}```
"""

promotion = """Nuggt allows you to create deployable AI apps with a single command. 
You can create an app for anything you imagine. For example, the following command would create an app that retrieves data from a PDF and creates an excel sheet for it.
```Open {upload:file} using {tool:document_loader}, fetch data for {text:query} and organise it in an excel file using {tool:excel_tool}``` 
You can create your own app at [Nuggt Playground](https://nuggt.io)
"""

user_input = """Step: Answer {text:query} by retrieving information from {upload:file} using {tool:document_tool}. 
"""

output_format = "Answer: Answer to the original query\nSource: The most relevant source in the document\nPage Number: The page number of the most relevant source."

st.subheader("Document QnA")
st.markdown("""Upload a PDF to retrieve information. Once you upload the PDF, you can simply type a question in the textbox to retrieve information and the relevant page number from the PDF. """)
st.markdown("""This app was created with a single AI command - 
            **Open :red[{upload : file}] using :green[{tool : document_loader}] and answer :blue[{text : query}] using :green[{tool : document_chat}]**  \n
""")
st.markdown("""**- :red[file]** is a variable that defines your input file.
""")
st.markdown("""**- :green[document_loader]** is a tool used for the loading uploaded documents.
""")
st.markdown("""**- :blue[query]** is a variable that defines your input query.
""")
st.markdown("""**- :green[document_chat]** is a tool that enables you to chat with your uploaded document.
""")

if user_input and output_format:
    save_to_sheets(user_input, output_format, "-", "-")
    variables = extract_variables(user_input)
    nuggt(user_input, output_format, variables)














