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
import streamlit as st

openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "9cae0f9d724d3cb2e51211d8e49dfbdc22ab279b"
os.environ["SCENEX_API_KEY"] = "f7GcmHvrJY050vmMn85L:1b7202dcbd71af619f044f87fc6721c5233c24e3cd64e2ee9c9ff69e29647024"
search_api = GoogleSerperAPIWrapper()

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
    global python_repl
    print(f"The code before changes:\n{code}")
    code = code.strip("```")
    code = code.strip("python\n")
    result = python_repl.run(code) 
    print(f"The code after changes:\n{code}")
    print(f"Output of the code:\n{result}")
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

def stablediffusion(query):
    prompt = StableDiffusionPromptGeneratorTool().langchain.run(query)
    #return StableDiffusionTool().langchain.run(query)
    return StableDiffusionTool().run(prompt)

def image_caption(path):
    return SceneXplainTool().run(path)

def generate_video(query):
    return TextToVideoTool().langchain.run(query)

def get_tools(variable_dictionary):
    tools = []
    tools_description = "\n\nYou can use the following tools:\n\n" 
    for variable in variable_dictionary.keys():
        if variable_dictionary[variable]["type"] == "tool":
            tools.append(variable_dictionary[variable]["name"])
            tools_description = tools_description + "Tool Name: " + variable_dictionary[variable]["name"] + "\nWhen To Use: " + variable_dictionary[variable]["use"] + "\nInput: " + variable_dictionary[variable]["input"]
            tools_description = tools_description + "\n\n"
    return (tools, tools_description)

def add_to_variable_dictionary(tool_name):
    tools = {
        "python": {"type": "tool", "name": "python", "use": "A Python shell. Use this to execute python code. Input should be a valid python code. If you want to see the output of a value, you should print it out with `print(...)`. Assume all packages are already installed.", "input": "Input should be a valid python code. Ensure proper indentation", "function": python},
        "search": {"type": "tool", "name": "search", "use": "Use this tool to get information from the internet", "input": "Input should be the query you want to search", "function": search},
        "video_tool": {"type": "tool", "name": "video_tool", "use": "useful when you want to retrieve information from a video", "input": "The input should be a JSON of the following format:\n{\"video_url\": \"URL of the video\", \"information\": \"the information you want to retrieve from the video\"}", "function": video_tool},
        "llm": {"type": "tool", "name": "llm", "use": "useful to get answer from an llm model", "input": "The input should be in the following format:\n{\"prompt\": \"The prompt to initialise the LLM\", \"input\": \"The input to the LLM\"}", "function": custom_llm},
        "stablediffusion": {"type": "tool", "name": "stablediffusion", "use": "Use this to generate an image from a prompt. This tool will return the path to the generated image.", "input": "the prompt to generate the image", "function": stablediffusion},
        "generate_video": {"type": "tool", "name": "generate_video", "use": "Use this to generate a video from a prompt. This tool will return the path to the generated video.", "input": "the prompt to generate the video", "function": generate_video},
        "image_caption": {"type": "tool", "name": "image_caption", "use": "Use this to caption an image.", "input": "the path to the image", "function": image_caption},
        }
    return tools[tool_name]

def get_variable_dictionary(variables):
    variable_dictionary = {}
    for index, variable in enumerate(variables):
        # type = st.text_input(f"Choose type for {variable}: ", key={str(index) + "_type"})
        type = st.radio(f"Choose type for {variable}: ", ('tool', 'video', 'user input'), key={str(index) + "_type"})
        if type == "user input":
            variable_dictionary[variable] = {"type": type}
        elif type == "video":
            video_url = st.text_input("Enter the video URL: ", key={str(index) + "_video"})
            variable_dictionary[variable] = {"type": type, "content": video_url}
            variable_dictionary["video_tool"] = add_to_variable_dictionary("video_tool")
        elif type == "tool":
            # tool_name = st.text_input("Choose a tool: ", key={str(index) + "_tool"})
            tool_name = st.radio(f"Choose a tool: ", ('python', 'search', 'video_tool', 'llm', 'stablediffusion', 'generate_video', 'image_caption'), key={str(index) + "_tool"})
            variable_dictionary[variable] = {"type": "tool_name", "content": tool_name}
            variable_dictionary[tool_name] = add_to_variable_dictionary(tool_name)
    return variable_dictionary

def get_nuggt(user_input, variable_dictionary):
    for variable in variable_dictionary.keys():
        if variable_dictionary[variable]["type"] == "user input":
            temp = st.text_input("Enter your input: ", key={variable + "temp"})
            replace_string = "{" + variable + "}"
            user_input = user_input.replace(replace_string, "<" + temp + ">")
        elif variable_dictionary[variable]["type"] == "video":
            replace_string = "{" + variable + "}"
            user_input = user_input.replace(replace_string, "<" + variable_dictionary[variable]["content"] + ">")
        elif variable_dictionary[variable]["type"] == "tool":
            tool_name = variable_dictionary[variable]["name"]
            replace_string = "{" + variable + "}"
            user_input = user_input.replace(replace_string, "<" + tool_name + ">")
            #tool_use = variable_dictionary[variable]["use"]
            #tool_input = variable_dictionary[variable]["input"]
            #user_input = user_input + "\n\nYou can use the following tools:\n\n" + "Tool Name: " + tool_name + "\nWhen To Use: " + tool_use + "\nInput: " + tool_input

    return user_input

def initialise_agent(nuggt, variable_dictionary):   
    messages = [{"role": "user", "content": nuggt}]
    output = ""
    while(True):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0, 
            stop=["\nObservation: "]
        )
        output = response.choices[0].message["content"]
        output = output.replace("Observation:", "")
        if "\nFinal Answer:" in output:
            # print(output)
            return output.split("Final Answer: ")[1]
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, output, re.DOTALL)
        if not match:
           print("I was here")
           output = "You are not following the format. Please follow the given format."
           messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]
           continue
           #raise ValueError(f"Could not parse LLM output: `{output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        observation = variable_dictionary[action]["function"](action_input)
        output = output + "\nObservation: " + observation + "\nThought: "
        print(output)
        messages = [{"role": "user", "content": messages[0]["content"] + "\n" + output}]
        #print(messages[0]["content"])

def get_output(output, output_type):
    output_map = {
        "acknowledgement": st.write("Acknowledged"),
        "image file": st.download_button("Download image", output),
        "video file": st.download_button("Download video", output),
        "txt file": st.download_button("Download text file", output),
        "code file": st.download_button("Download code file", output),
    }
    return output_map[output_type]

    
def main():
    st.title('Nuggt.io')

    # UI for entering user_input

    with st.expander("See how it works"):
        st.write("The chart above shows some numbers I picked for you. I rolled actual dice for these, so they are *guaranteed* to be random.")

    user_input = st.text_input("Enter your instruction: ", key="enter_instruction")
    if user_input:
        # Process user_input
        variables = extract_variables(user_input)
        variable_dictionary = get_variable_dictionary(variables)
        nuggt = get_nuggt(user_input, variable_dictionary)
        tools, tools_description = get_tools(variable_dictionary)
        # output = st.text_input("Enter output format: ", key="output")
        output = st.radio(f"Enter output format: ", ('acknowledgement', 'video file', 'code file', "image file", "txt file"), key="output")

        output_format = f"""\nUse the following format:
        Thought: you should always think about what to do
        Action: the action to take, should be one of {tools}.
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: {output}
        """
        nuggt = nuggt + tools_description + output_format
        # st.text(nuggt)
        # st.text(initialise_agent(nuggt, variable_dictionary))
        get_output(initialise_agent(nuggt, variable_dictionary), output)

if __name__ == "__main__":
    main()

