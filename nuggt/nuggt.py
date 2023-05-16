from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.utilities import PythonREPL, GoogleSerperAPIWrapper
from gradio_tools.tools import (StableDiffusionTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool, TextToVideoTool)
from langchain.tools import SceneXplainTool
from io import StringIO
import traceback
import openai
import sys
import json
import re
import os
import glob
import streamlit as st
import gspread
from google.oauth2 import service_account
import pandas as pd
from shillelagh.backends.apsw.db import connect

st.set_page_config(page_title="Nuggt", layout="wide")
count = 0
openai.api_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["SERPER_API_KEY"] = st.secrets["serper_api_key"]
os.environ["SCENEX_API_KEY"] = st.secrets["scenex_api_key"]
search_api = GoogleSerperAPIWrapper()
# scopes = [
#     "https://www.googleapis.com/auth/spreadsheets",
# ]
# skey = st.secrets["gcp_service_account"]
# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets["gcp_service_account"],
#     scopes=[
#         "https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"
#     ],
# )
# conn = connect(credentials=credentials)
# client = gspread.authorize(credentials)

# sheet_id = '1oh5koIOI35pZeKZEPpKTW3aPiOIVxaVasj5d2YXx7w8'
# csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
# database_df = pd.read_csv(csv_url, on_bad_lines='skip')

# database_df = database_df.astype(str)
# sheet_url = st.secrets["private_gsheets_url"] #this information should be included in streamlit secret
# sheet = client.open_by_url(sheet_url).sheet1
# sheet.update([database_df.columns.values.tolist()] + database_df.values.tolist())
# st.success('Data has been written to Google Sheets')

sheet_url = st.secrets["private_gsheets_url"]

def create_connection():
    #  if 'email' in st.session_state:
    credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], 
    scopes=["https://www.googleapis.com/auth/spreadsheets",],)
    connection = connect(":memory:", adapter_kwargs={
        "gsheetsapi" : { 
        "service_account_info" : {
            "type" : st.secrets["gcp_service_account"]["type"],
            "project_id" : st.secrets["gcp_service_account"]["project_id"],
            "private_key_id" : st.secrets["gcp_service_account"]["private_key_id"],
            "private_key" : st.secrets["gcp_service_account"]["private_key"],
            "client_email" : st.secrets["gcp_service_account"]["client_email"],
            "client_id" : st.secrets["gcp_service_account"]["client_id"],
            "auth_uri" : st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri" : st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url" : st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url" : st.secrets["gcp_service_account"]["client_x509_cert_url"],
            }
        },
    })
    return connection.cursor()

def execute_query(query):
    cursor = create_connection()
    rows = cursor.execute(query)
    rows = rows.fetchall()

class PythonREPLa:
    def __init__(self):
        self.local_vars = {}

    def run(self, code: str) -> str:
        # Redirect stdout and stderr to StringIO
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = sys.stdout = StringIO()
        redirected_error = sys.stderr = StringIO()

        try:
            exec(code, self.local_vars)
        except Exception:
            traceback.print_exc()

        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Get the output or error message
        output = redirected_output.getvalue()
        error = redirected_error.getvalue()

        return output if not error else "Your code has the following error. Please provide the corrected code.\n" + error

python_repl = PythonREPLa()

def extract_code_from_block(text):
    if "!pip" in text:
        return "The package is successfully installed."
    pattern = r'```python\n(.*?)\n```'
    code = re.search(pattern, text, re.DOTALL)
    
    if code:
        return code.group(1).strip()
    else:
        return text

def extract_variables(input_string):
    pattern = r'\{(.*?)\}'
    variables = re.findall(pattern, input_string)
    return variables

def load_video(video_url):
    embeddings = OpenAIEmbeddings()
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def video_tool(query):
    data = json.loads(query)
    transcript = load_video(data["video_url"])
    docs = transcript.similarity_search(data["information"])
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain.run(input_documents=docs, question=query)

def python(code):
    code = extract_code_from_block(code)
    result = python_repl.run(code) 
    #print(f"The code after changes:\n{code}")
    #print(f"Output of the code:\n{result}")
    return result  

def display(code):
    code = extract_code_from_block(code)
    result = python_repl.run(code) 
    #print(f"The code after changes:\n{code}")
    #print(f"Output of the code:\n{result}")
    return result  

def search(query):
    global search_api
    return search_api.run(query)

def custom_llm(query):
    data = json.loads(query)
    messages = [
        {"role": "system", "content": data["prompt"]},
        {"role": "user", "content": data["input"]}
    ]
    response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0, 
    )

    return response.choices[0].message["content"].strip()

def stable_diffusion(query):
    prompt = StableDiffusionPromptGeneratorTool().langchain.run(query)
    #return StableDiffusionTool().langchain.run(query)
    return StableDiffusionTool().run(prompt)

def image_caption(path):
    return SceneXplainTool().run(path)

def generate_video(query):
    return TextToVideoTool().langchain.run(query)

def human(query):
    pass
    

def get_tool_info(tool_name):
    tools = {
        "python": {"type": "tool", "name": "python", "use": f"A Python shell. Use this to execute python code. Input should be a valid python code. If you want to see the output of a value, you should print it out with `print(...)`. Assume that the package is already installed. You can try pip install package-name", "input": "Input should be a valid python code. Ensure proper indentation", "function": python},
        "search": {"type": "tool", "name": "search", "use": "Use this tool to get information from the internet", "input": "Input should be the query you want to search", "function": search},
        "video_tool": {"type": "tool", "name": "video_tool", "use": "useful when you want to retrieve information from a video", "input": "The input should be a JSON of the following format:\n{\"video_url\": \"URL of the video\", \"information\": \"the information you want to retrieve from the video\"}", "function": video_tool},
        "llm": {"type": "tool", "name": "llm", "use": "useful to get answer from an llm model", "input": "The input should be in the following format:\n{\"prompt\": \"The prompt to initialise the LLM\", \"input\": \"The input to the LLM\"}", "function": custom_llm},
        "stable_diffusion": {"type": "tool", "name": "stable_diffusion", "use": "Use this to generate an image from a prompt. This tool will return the path to the generated image.", "input": "the prompt to generate the image", "function": stable_diffusion},
        "generate_video": {"type": "tool", "name": "generate_video", "use": "Use this to generate a video from a prompt. This tool will return the path to the generated video.", "input": "the prompt to generate the video", "function": generate_video},
        "image_caption": {"type": "tool", "name": "image_caption", "use": "Use this to caption an image.", "input": "the path to the image", "function": image_caption},
        "display" : {"type": "tool", "name": "display", "use": "Use this to display things using streamlit", "input": "The input should be a valid python code using the streamlit library", "function": display},
        "human": {"type": "tool", "name": "human", "use": "Use this to get input from the user", "input": "The input should be the information you want from the user.", "function": human},
        }
    return tools[tool_name]

def nuggt(user_input, output_format, variables):
    tools = []
    tools_description = "\n\nYou can use the following tools:\n\n" 
    value_dict = {}
    form_user = st.form("user-form")
    output_type = ""
    for variable in variables:
        type = variable.split(":")[0]
        choice = variable.split(":")[1]
        if type == "text":
            if choice not in value_dict.keys():
                temp = form_user.text_input(f"Enter value for {choice}: ")

                replace_string = "{" + variable + "}"
                user_input = user_input.replace(replace_string, "<" + temp + ">")
                value_dict[choice] = temp
            else:
                replace_string = "{" + variable + "}"
                user_input = user_input.replace(replace_string, "<" + value_dict[choice] + ">")

        elif type == "upload":
            if choice not in value_dict.keys():
                uploaded_file = form_user.file_uploader(f"Upload {choice}") 
                replace_string = "{" + variable + "}"
                if uploaded_file:
                    user_input = user_input.replace(replace_string, "<" + uploaded_file.name + ">")
                    value_dict[choice] = uploaded_file.name
            else:
                replace_string = "{" + variable + "}"
                user_input = user_input.replace(replace_string, "<" + value_dict[choice] + ">")

        elif type == "tool":
            replace_string = "{" + variable + "}"
            user_input = user_input.replace(replace_string, "<" + choice + ">")
            if choice not in value_dict.keys():
                tools.append(choice)
                tool_info = get_tool_info(choice)
                tools_description = tools_description + "Tool Name: " + tool_info["name"] + "\nWhen To Use: " + tool_info["use"] + "\nInput: " + tool_info["input"]
                tools_description = tools_description + "\n\n"
                value_dict[choice] = tool_info["function"]
    
    agent_instruction = f"""\nUse the following format:
        Thought: you should always think about what to do
        Action: the action to take, should be one of {tools}.
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: {output_format}
        """
    nuggt = user_input + tools_description + agent_instruction
    submit = form_user.form_submit_button("Submit")
    if submit:
        st.write(initialise_agent(nuggt, value_dict))
        
def initialise_agent(nuggt, value_dict):   
    messages = [{"role": "user", "content": nuggt}]
    output = ""
    log_expander = st.expander('Logs')  # create expander
    while(True):
        print("agent called")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0, 
            stop=["\nObservation: "]
        )
        output = response.choices[0].message["content"]
        output = output.replace("Observation:", "")
        with log_expander:  # write to expander
            st.write(output + "\n")

        if "\nFinal Answer:" in output:
            print(output)
            return output.split("Final Answer: ")[1]
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, output, re.DOTALL)
        if not match:
            print("I was here")
            #print(output)
            output = "You are not following the format. Please follow the given format."
            messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]
            continue
        #raise ValueError(f"Could not parse LLM output: `{output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        observation = value_dict[action](action_input)
        # st.write(f"Observation: {observation}\n")
        output = output + "\nObservation: " + observation + "\nThought: "
        messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]
        print(messages[0]["content"])
        print("---"*30)

def get_most_recent_file(dir_path):
    # Get a list of all files in directory
    files = glob.glob(dir_path + "/*")
    
    if not files:
        return None
    
    # Find the most recent file
    most_recent_file = max(files, key=os.path.getctime)

    return most_recent_file

def main():
    st.title('Nuggt.io')

    query = "INSERT INTO user_input (Column1, Column2, Column3) VALUES ('Value1', 'Value2', 'Value3');"
    execute_query(query)

    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.subheader("How to use")

        st.markdown("""
        **Step 1:** Write your instruction. Below are variables provided:

        - **Define user input:** 
            - `{text: variable name}`. Eg, `{text:stock_name}`, `{text:youtube_url}`
            - `{upload: file name}`. Eg, `{upload: annual_report.pdf}`, `{upload: main.py}`

        - **Define tools:** 
            - `{tool: tool_name}`. Eg, python, search, video_tool, llm, stable_diffusion, generate_video, image_caption

        **Step 2:** Define your output format to be an "Acknowledgement".

        **Step 3:** Generate your application. 

        **Step 4:** Use your application by providing it input and downloading your output.
        """)

        st.markdown("---")

        st.subheader("About")
        st.markdown("""
        **Nuggt** allows you to build and share end-to-end applications using large language model. As we are currently in alpha, we appreciate feedback to improve the platform and the responses.
        """)

        st.markdown("---")

        st.subheader("Use Cases")

        st.markdown("""
        **Create a finance bot to get Moving Average Convergence/Divergence indicator over the last 7 days**

        **Instructions:** Using yfinance api, which is already installed, pull the last 7 days OHLC data for `{text:stock_name}` using `{tool:python}`, calculate the MACD and save only the MACD to a text file called `{text:file_name}` in the project repository using `{tool:python}`. If you are stuck in a loop, try another strategy.

        **Input:** stock_name is **TSLA**, file_name is **tsla_macd.txt**.

        **Create an image bot that takes in an image as inspiration and generates variations of it using stable diffusion**

        **Instructions:** Take in an image: `{upload:inspiration_image}` using `{tool:python}` and come up with a creative prompt that matches the contents of the image. Then use `{tool:stable_diffusion}` to generate an image save it as `{text:file_name}`.

        **Input:** inspiration_image is a **.png file** that you upload, file_name is **generated_image.png**.
        """)

    with col2:
        user_input = st.text_area("Enter your instruction: ", key="enter_instruction")
        output_format = st.text_input("Enter output format: ", key="output")

        if user_input and output_format:
            variables = extract_variables(user_input)
            nuggt(user_input, output_format, variables)

            """most_recent_file = get_most_recent_file("path_to_your_repo")

            # Provide a download button for the file
            with open(most_recent_file, "rb") as file:
                file_content = file.read()

            st.download_button(
                label="Download file",
                data=file_content,
                file_name=os.path.basename(most_recent_file),
                mime="application/octet-stream",
            )"""
        
if __name__ == "__main__":
    main()

