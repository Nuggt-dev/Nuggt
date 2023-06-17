import glob
import os
import re
import tempfile

import openai
import requests
import streamlit as st
import toml
import tool
from colorama import Fore
from dotenv import load_dotenv
from helper.sidebar_functions import sidebar_logo

load_dotenv()
config = toml.load("./.streamlit/config.toml")

primary_color = config["theme"]["primaryColor"]
secondary_background_color = config["theme"]["secondaryBackgroundColor"]
text_color = config["theme"]["textColor"]

st.markdown(
    f"""
    <style>
        :root {{
            --primary-color: {primary_color};
            --secondary-background-color: {secondary_background_color};
            --text-color: {text_color};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

sidebar_logo("assets/nuggt-logo.png")

count = 0
global tmp_path


def save_to_sheets(userInput, outputFormat, feedback, logs):
    url = "https://docs.google.com/forms/d/1PveqD5klH2geQvI3nlkI6l-chBctNz6O-jmpwSO2FYk/formResponse"

    data = {
        "entry.2013000889": userInput,
        "entry.586411750": outputFormat,
        "entry.1340987871": feedback,
        "entry.697215161": logs,
    }
    try:
        requests.post(url, data=data)
    except:
        print("Error!")


def is_file(filename):
    return os.path.isfile(filename)


def extract_variables(input_string):
    pattern = r"\{(.*?)\}"
    variables = re.findall(pattern, input_string)
    return variables


def get_tool_info(tool_name):
    tools = {
        "python": {
            "type": "tool",
            "name": "python",
            "use": "Use this to execute python code. Display your results using the print funçction.",
            "input": "Input should be a valid python code. Ensure proper indentation",
            "function": tool.python,
        },
        "search": {
            "type": "tool",
            "name": "search",
            "use": "Use this tool to get information from the internet",
            "input": "Input should be the query you want to search",
            "function": tool.search,
        },
        "video_tool": {
            "type": "tool",
            "name": "video_tool",
            "use": "useful when you want to retrieve information from a video",
            "input": 'The input should be a JSON of the following format:\n{"video_url": "URL of the video", "information": "the information you want to retrieve from the video"}',
            "function": tool.video_tool,
        },
        "llm": {
            "type": "tool",
            "name": "llm",
            "use": "useful to get answer from an llm model",
            "input": 'The input should be in the following format:\n{"prompt": "The prompt to initialise the LLM", "input": "The input to the LLM"}',
            "function": tool.custom_llm,
        },
        "stable_diffusion": {
            "type": "tool",
            "name": "stable_diffusion",
            "use": "Use this to generate an image from a prompt. This tool will return the path to the generated image.",
            "input": "the prompt to generate the image",
            "function": tool.stable_diffusion,
        },
        "generate_video": {
            "type": "tool",
            "name": "generate_video",
            "use": "Use this to generate a video from a prompt. This tool will return the path to the generated video.",
            "input": "the prompt to generate the video",
            "function": tool.generate_video,
        },
        "image_caption": {
            "type": "tool",
            "name": "image_caption",
            "use": "Use this to caption an image.",
            "input": "the path to the image",
            "function": tool.image_caption,
        },
        "display": {
            "type": "tool",
            "name": "display",
            "use": "Use this to display things using streamlit",
            "input": "The input should be a valid python code using the streamlit library",
            "function": tool.display,
        },
        "browse_website": {
            "type": "tool",
            "name": "browse_website",
            "use": "Use this to get information from a website",
            "input": 'The input should be in the following format:\n{"url": "URL of the website", "information": "the information you want to retrieve from the website"}',
            "function": tool.browse_website,
        },
        "google": {
            "type": "tool",
            "name": "google",
            "use": "use it to get google results",
            "input": "The input should be a google query",
            "function": tool.google,
        },
        "document_tool": {
            "type": "tool",
            "name": "document_tool",
            "use": "useful when you want to retrieve information from a document",
            "input": 'The input should be a JSON of the following format:\n{"document_name": "name of the document you want to retrieve information from", "information": "the information you want to retrieve from the document"}',
            "function": tool.document_tool,
        },
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

                if is_file(temp):
                    new_file_path = os.path.join(os.getcwd(), temp)
                    print(new_file_path)

                replace_string = "{" + variable + "}"
                user_input = user_input.replace(replace_string, "<" + temp + ">")
                value_dict[choice] = temp
            else:
                replace_string = "{" + variable + "}"
                user_input = user_input.replace(
                    replace_string, "<" + value_dict[choice] + ">"
                )

        elif type == "upload":
            if choice not in value_dict.keys():
                uploaded_file = form_user.file_uploader(f"Upload {choice}")
                replace_string = "{" + variable + "}"
                if uploaded_file:
                    user_input = user_input.replace(
                        replace_string, "<" + uploaded_file.name + ">"
                    )
                    value_dict[choice] = uploaded_file.name
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        global tmp_path
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    with open(os.path.join("", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())

            else:
                replace_string = "{" + variable + "}"
                user_input = user_input.replace(
                    replace_string, "<" + value_dict[choice] + ">"
                )

        elif type == "tool":
            replace_string = "{" + variable + "}"
            user_input = user_input.replace(
                replace_string, "<" + "action " + choice + ">"
            )
            if choice not in value_dict.keys():
                tools.append(choice)
                tool_info = get_tool_info(choice)
                tools_description = (
                    tools_description
                    + "Action Name: "
                    + tool_info["name"]
                    + "\nWhen To Use: "
                    + tool_info["use"]
                    + "\nInput: "
                    + tool_info["input"]
                )
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
    print(nuggt)
    submit = form_user.form_submit_button("Submit")
    if submit:
        with st.spinner("I am still working on it...."):
            try:
                agent = st.write(initialise_agent(nuggt, value_dict, tools))
            except Exception as e:
                print(e)
                st.write(
                    "Oops, something went wrong. Please make sure you have entered the correct API keys. If the issue still persists, please reach out to us at https://discord.gg/gzdCDM84"
                )
        # save_to_sheets("-", "-", "-", agent)
        feedback = st.text_input(
            "Thank you for experimenting with Nuggt! We would appreciate some feedback to help improve the product :smiley:"
        )
        save_to_sheets("-", "-", feedback, "-")
        if uploaded_file:
            os.remove(uploaded_file.name)
        # st.write("All uploaded files have been deleted")


def initialise_agent(nuggt, value_dict, tools):
    messages = [{"role": "user", "content": nuggt}]
    output = ""
    count = 0
    log_expander = st.expander("Logs")  # create expander
    while True:
        count = count + 1
        if count > 10:
            raise ValueError("Too many steps")

        response = openai.ChatCompletion.create(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.getenv("MODEL_NAME"),
            messages=messages,
            temperature=0,
            top_p=0,
            stop=["\nObservation: "],
        )

        output = response.choices[0].message["content"]
        output = output.replace("Observation:", "")
        print(Fore.BLUE + output)

        # with log_expander:  # write to expander
        #    st.write(output.split("Reason:")[0] + "\n")

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, output, re.DOTALL)

        if "Final Answer:" in output and not match:
            return output.split("Final Answer:")[1]
            # return output.replace("Final Answer:", "")

        if "Step" not in output:
            print(Fore.YELLOW + "The model didnt output a step.")
            output = (
                "Please follow the format Step/Reason/Action/Action Input/Observation"
            )
            messages = [
                {"role": "user", "content": messages[0]["content"] + "\n" + output}
            ]
            continue

        if "Reason" not in output:
            print(Fore.YELLOW + "The model didnt output a reason.")
            output = (
                "Please follow the format Step/Reason/Action/Action Input/Observation"
            )
            messages = [
                {"role": "user", "content": messages[0]["content"] + "\n" + output}
            ]
            continue

        if output.count("Action Input") > 1:
            print(Fore.YELLOW + "The model went crazy.")
            output = "Please go one step at a time."
            messages = [
                {"role": "user", "content": messages[0]["content"] + "\n" + output}
            ]
            continue

        if not match:
            print(Fore.RED + "The model was sidetracked.")
            output = "You are not following the format. Please follow the given format."
            messages = [
                {"role": "user", "content": messages[0]["content"] + "\n" + output}
            ]
            continue

        action = match.group(1).strip().lower()
        if action not in tools:
            output = f"Invalid Action. Your action should be one of {tools}."
            print(Fore.YELLOW + "The agent forgot his tools." + output)
            messages = [
                {"role": "user", "content": messages[0]["content"] + "\n" + output}
            ]
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
    st.header("Nuggt Playground")
    user_input = st.text_area(label="Enter instruction here")

    output_format = st.text_input(label="Output format")

    if user_input and output_format:
        # save_to_sheets(user_input, output_format, "-", "-")
        variables = extract_variables(user_input)
        nuggt(user_input, output_format, variables)


if __name__ == "__main__":
    main()
