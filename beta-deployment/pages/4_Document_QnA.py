from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper, GoogleSearchAPIWrapper
from gradio_tools.tools import (StableDiffusionTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool, TextToVideoTool)
from langchain.tools import SceneXplainTool
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader, SeleniumURLLoader
from langchain.chains import RetrievalQA
from io import StringIO
import traceback
import openai
import sys
import json
import re
import os
import glob
import streamlit as st
import browse
import requests
from colorama import Fore
import tempfile

model_name = "gpt-3.5-turbo"
st.set_page_config(page_title="Nuggt", layout="wide")
count = 0
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "9cae0f9d724d3cb2e51211d8e49dfbdc22ab279b"
os.environ["SCENEX_API_KEY"] = "f7GcmHvrJY050vmMn85L:1b7202dcbd71af619f044f87fc6721c5233c24e3cd64e2ee9c9ff69e29647024"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBq3rhqM03-hqLbDqHeKHbc8K2qSgqMW7Q"
os.environ["GOOGLE_CSE_ID"] = "a60ee944812a441d9"

search_api = GoogleSerperAPIWrapper()
global tmp_path
def save_to_sheets(userInput, outputFormat, feedback, logs):
    url = "https://docs.google.com/forms/d/1PveqD5klH2geQvI3nlkI6l-chBctNz6O-jmpwSO2FYk/formResponse"

    data = {
        'entry.2013000889': userInput,
        'entry.586411750': outputFormat,
        'entry.1340987871': feedback,
        'entry.697215161': logs
    }
    try:
        requests.post(url, data = data)
    except:
        print("Error!")

def is_file(filename):
    return os.path.isfile(filename)

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
        if "Traceback" not in error:
            error = False

        return output if not error else "Your code has the following error. Please provide the corrected code.\n" + error

python_repl = PythonREPLa()

def extract_code_from_block(text):
    if "!pip" in text:
        return "The package is successfully installed."
    text = text.replace("`", "")
    text = text.replace("python", "")
    return text

def google(query):
    search = GoogleSearchAPIWrapper()
    return str(search.results(query.replace('"', ''), 3))

def browse_website(query):
    data = json.loads(query)
    text = browse.scrape_text(data["url"])
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(text)
    db = FAISS.from_texts(docs, embeddings)
    docs = db.similarity_search(data["information"])
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain.run(input_documents=docs, question=data["information"])

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

def document_tool(query):
    data = json.loads(query)
    
    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k":2})
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                chain_type="stuff", 
                                retriever=retriever, 
                                return_source_documents=True,
                                )
    return str(rqa(data["information"]))
    
    
    docs = faiss_index.similarity_search(data["information"], k=3)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=data["information"])
    content = ""
    for doc in docs:
        content = content + str(doc.metadata["page"]) + ":" + doc.page_content[:300] + "\n"
    return content
    """transcript = PdfReader(tmp_path)
    raw_text = ""
    for i, page in enumerate(transcript.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    
    text_splitter = RecursiveCharacterTextSplitter(        
        #separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 10,
        #length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    docs = docsearch.similarity_search(data["information"])
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=data["information"])"""

def python(code):
    code = extract_code_from_block(code)
    result = python_repl.run(code) 
    if "Your code has the following error." in result:
        result = fix_error(code, result)

    if result == "":
        result = "Your code was successfully executed."

    return result  

def fix_error(code, result):
    while "Your code has the following error." in result:
        error = result.replace("Your code has the following error. Please provide the corrected code.", "")
        user_input = f"Input:\n{code}\nError:\n{error}\nOutput:"

        print(f"I am going to correct: {user_input}")
        print(Fore.RED + "Code needs some correction.")
        messages = [
        {"role": "user", "content": f"""Output the corrected code in the following format:\n```Your code here```\n{user_input}"""},
        ]

        response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=0, 
        )

        code = response.choices[0].message["content"].strip()
        code = code.replace("`", "")
        print(f"\nOutput of the model: {code}")

        result = python_repl.run(code)
        

    print(Fore.GREEN + "Code has been corrected.")
    return result

def display(code):
    code = extract_code_from_block(code)
    result = python_repl.run(code) 
    if "Your code has the following error." in result:
        result = fix_error(code, result)

    if result == "":
        result = "Your code was successfully executed."
        
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
            model=model_name,
            messages=messages,
            temperature=0, 
    )

    return response.choices[0].message["content"].strip()

def stable_diffusion(query):
    prompt = StableDiffusionPromptGeneratorTool().langchain.run(query)
    return StableDiffusionTool().run(prompt)

def image_caption(path):
    return SceneXplainTool().run(path)

def generate_video(query):
    return TextToVideoTool().langchain.run(query)

def human(query):
    pass

def get_tool_info(tool_name):
    tools = {
        "python": {"type": "tool", "name": "python", "use": "Use this to execute python code. Display your results using the print funçction.", "input": "Input should be a valid python code. Ensure proper indentation", "function": python},
        "search": {"type": "tool", "name": "search", "use": "Use this tool to get information from the internet", "input": "Input should be the query you want to search", "function": search},
        "video_tool": {"type": "tool", "name": "video_tool", "use": "useful when you want to retrieve information from a video", "input": "The input should be a JSON of the following format:\n{\"video_url\": \"URL of the video\", \"information\": \"the information you want to retrieve from the video\"}", "function": video_tool},
        "llm": {"type": "tool", "name": "llm", "use": "useful to get answer from an llm model", "input": "The input should be in the following format:\n{\"prompt\": \"The prompt to initialise the LLM\", \"input\": \"The input to the LLM\"}", "function": custom_llm},
        "stable_diffusion": {"type": "tool", "name": "stable_diffusion", "use": "Use this to generate an image from a prompt. This tool will return the path to the generated image.", "input": "the prompt to generate the image", "function": stable_diffusion},
        "generate_video": {"type": "tool", "name": "generate_video", "use": "Use this to generate a video from a prompt. This tool will return the path to the generated video.", "input": "the prompt to generate the video", "function": generate_video},
        "image_caption": {"type": "tool", "name": "image_caption", "use": "Use this to caption an image.", "input": "the path to the image", "function": image_caption},
        "display" : {"type": "tool", "name": "display", "use": "Use this to display things using streamlit", "input": "The input should be a valid python code using the streamlit library", "function": display},
        "human": {"type": "tool", "name": "human", "use": "Use this to get input from the user", "input": "The input should be the information you want from the user.", "function": human},
        "browse_website": {"type": "tool", "name": "browse_website", "use": "Use this to get information from a website", "input": "The input should be in the following format:\n{\"url\": \"URL of the website\", \"information\": \"the information you want to retrieve from the website\"}", "function": browse_website},
        "google": {"type": "tool", "name": "google", "use": "use it to get google results", "input": "The input should be a google query", "function": google},
        "document_tool": {"type": "tool", "name": "document_tool", "use": "useful when you want to retrieve information from a document", "input": "The input should be a JSON of the following format:\n{\"document_name\": \"name of the document you want to retrieve information from\", \"information\": \"the information you want to retrieve from the document\"}", "function": document_tool},
        }
    return tools[tool_name]

def nuggt(user_input, output_format, variables):
    tools = []
    tools_description = "\n\nYou can use the following actions:\n\n" 
    value_dict = {}
    uploaded_file = None
    form_user = st.form("user-form")
    for variable in variables:
        type = variable.split(":")[0]
        choice = variable.split(":")[1]
        if type == "text":
            if choice not in value_dict.keys():
                temp = form_user.text_input(f"Enter value for {choice}: ")

                if (is_file(temp)):
                    new_file_path = os.path.join(os.getcwd(), temp)
                    print(new_file_path)

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
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        global tmp_path
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    with open(os.path.join("",uploaded_file.name),"wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
            else:
                replace_string = "{" + variable + "}"
                user_input = user_input.replace(replace_string, "<" + value_dict[choice] + ">")

        elif type == "tool":
            replace_string = "{" + variable + "}"
            user_input = user_input.replace(replace_string, "<" + "action " + choice + ">")
            if choice not in value_dict.keys():
                tools.append(choice)
                tool_info = get_tool_info(choice)
                tools_description = tools_description + "Action Name: " + tool_info["name"] + "\nWhen To Use: " + tool_info["use"] + "\nInput: " + tool_info["input"]
                tools_description = tools_description + "\n\n"
                value_dict[choice] = tool_info["function"]
    
    agent_instruction = f"""\nUse the following format:
        Step 1: The first step
        Reason: Reason for taking this step
        Action: the action to take, should be one of {tools}.
        Action Input: the input to the action
        Observation: the result of the action
        
        Step 2: The second step
        Reason: Reason for taking this step
        Action: the action to take, should be one of {tools}.
        Action Input: the input to the action 
        Observation: the result of the action

        ... (this Step/Reason/Action/Action Input/Observation repeats for all steps)
        
        Once you have completed all the steps, your final answer should be in the format:
        Final Answer: {output_format}
        """
    nuggt = user_input + tools_description + agent_instruction
    submit = form_user.form_submit_button("Submit")
    if submit:
        with st.spinner('I am still working on it....'):
            try:
                agent = st.write(initialise_agent(nuggt, value_dict, tools))
            except:
                st.write("Our servers appear to be experiencing high traffic at the moment. Given that we're currently in our Beta phase, we sincerely appreciate your patience and understanding. Please try again in a short while. Thank you for your support during this exciting stage of our development!")
        save_to_sheets("-", "-", "-", agent)
        feedback = st.text_input("Thank you for experimenting with Nuggt! We would appreciate some feedback to help improve the product :smiley:")
        save_to_sheets("-", "-", feedback, "-")
        if uploaded_file:
            os.remove(uploaded_file.name)
        #st.write("All uploaded files have been deleted")

        
def initialise_agent(nuggt, value_dict, tools):   
    messages = [{"role": "user", "content": nuggt}]
    output = ""
    count = 0
    log_expander = st.expander('Logs')  # create expander
    while(True):
        count = count + 1
        if count > 10:
            raise ValueError("Too many steps")
        
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0, 
            stop=["\nObservation: "]
        )
       
        output = response.choices[0].message["content"]
        output = output.replace("Observation:", "")
        print(Fore.BLUE + output)

        #with log_expander:  # write to expander
        #    st.write(output.split("Reason:")[0] + "\n")

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, output, re.DOTALL)

        if "Final Answer:" in output and not match:
            return output.split("Final Answer:")[1]
            #return output.replace("Final Answer:", "")
        
        if "Step" not in output:
            print(Fore.YELLOW + "The model didnt output a step.")
            output = "Please follow the format Step/Reason/Action/Action Input/Observation"
            messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]
            continue
        
        if "Reason" not in output:
            print(Fore.YELLOW + "The model didnt output a reason.")
            output = "Please follow the format Step/Reason/Action/Action Input/Observation"
            messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]
            continue
        
        if output.count("Action Input") > 1:
            print(Fore.YELLOW + "The model went crazy.")
            output = "Please go one step at a time."
            messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]
            continue

        if not match:
            print(Fore.RED + "The model was sidetracked.")
            output = "You are not following the format. Please follow the given format."
            messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]
            continue
       
        action = match.group(1).strip()
        if action not in tools:
            output = f"Invalid Action. Your action should be one of {tools}."
            print(Fore.YELLOW + "The agent forgot his tools." + output) 
            messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]
            continue

        with log_expander:  # write to expander
            st.write(output.split("Reason:")[0] + "\n")

        action_input = match.group(2)
        observation = value_dict[action](action_input)
        print(Fore.GREEN + "\nObservation: " + observation)
        output = output + "\nObservation: " + observation 
        messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]

def get_most_recent_file(dir_path):
    # Get a list of all files in directory
    files = glob.glob(dir_path + "/*")
    
    if not files:
        return None
    
    # Find the most recent file
    most_recent_file = max(files, key=os.path.getctime)

    return most_recent_file


    
def main():

    guide = ""
    information = ""
    promotion = ""
    col1, col2 = st.columns([2, 3], gap="large")
    
    guide = """Upload a PDF to retrieve information. Once you upload the PDF, you can simply type a question in the textbox to retrieve information and the relevant page number from the PDF. """
    
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

    with col1:
            st.subheader("How To Use")
            st.markdown(guide)
            st.subheader("How Was This Created?")
            st.markdown(information)
            st.subheader("Create Your Own App at Nuggt Playground")
            st.markdown(promotion)
            
    with col2:
        if user_input and output_format:
            save_to_sheets(user_input, output_format, "-", "-")
            variables = extract_variables(user_input)
            nuggt(user_input, output_format, variables)

        
if __name__ == "__main__":
    main()














