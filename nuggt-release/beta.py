from streamlit_option_menu import option_menu
import openai
import re
import os
import glob
import streamlit as st
import requests
from colorama import Fore
import tempfile
from dotenv import load_dotenv
import tool
import openai

st.set_page_config(page_title="Nuggt", layout="wide")
load_dotenv()
count = 0
global tmp_path
model_name = os.getenv("MODEL_NAME")
openai.api_key = os.environ.get("OPENAI_API_KEY")

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
    pass

def extract_variables(input_string):
    pattern = r'\{(.*?)\}'
    variables = re.findall(pattern, input_string)
    return variables

def get_tool_info(tool_name):
    tools = {
        "python": {"type": "tool", "name": "python", "use": "Use this to execute python code. Display your results using the print funçction.", "input": "Input should be a valid python code. Ensure proper indentation", "function": tool.python},
        "search": {"type": "tool", "name": "search", "use": "Use this tool to get information from the internet", "input": "Input should be the query you want to search", "function": tool.search},
        "video_tool": {"type": "tool", "name": "video_tool", "use": "useful when you want to retrieve information from a video", "input": "The input should be a JSON of the following format:\n{\"video_url\": \"URL of the video\", \"information\": \"the information you want to retrieve from the video\"}", "function": tool.video_tool},
        "llm": {"type": "tool", "name": "llm", "use": "useful to get answer from an llm model", "input": "The input should be in the following format:\n{\"prompt\": \"The prompt to initialise the LLM\", \"input\": \"The input to the LLM\"}", "function": tool.custom_llm},
        "stable_diffusion": {"type": "tool", "name": "stable_diffusion", "use": "Use this to generate an image from a prompt. This tool will return the path to the generated image.", "input": "the prompt to generate the image", "function": tool.stable_diffusion},
        "generate_video": {"type": "tool", "name": "generate_video", "use": "Use this to generate a video from a prompt. This tool will return the path to the generated video.", "input": "the prompt to generate the video", "function": tool.generate_video},
        "image_caption": {"type": "tool", "name": "image_caption", "use": "Use this to caption an image.", "input": "the path to the image", "function": tool.image_caption},
        "display" : {"type": "tool", "name": "display", "use": "Use this to display things using streamlit", "input": "The input should be a valid python code using the streamlit library", "function": tool.display},
        "browse_website": {"type": "tool", "name": "browse_website", "use": "Use this to get information from a website", "input": "The input should be in the following format:\n{\"url\": \"URL of the website\", \"information\": \"the information you want to retrieve from the website\"}", "function": tool.browse_website},
        "google": {"type": "tool", "name": "google", "use": "use it to get google results", "input": "The input should be a google query", "function": tool.google},
        "document_tool": {"type": "tool", "name": "document_tool", "use": "useful when you want to retrieve information from a document", "input": "The input should be a JSON of the following format:\n{\"document_name\": \"name of the document you want to retrieve information from\", \"information\": \"the information you want to retrieve from the document\"}", "function": tool.document_tool},
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
    print(nuggt)
    submit = form_user.form_submit_button("Submit")
    if submit:
        with st.spinner('I am still working on it....'):
            try:
                agent = st.write(initialise_agent(nuggt, value_dict, tools))
            except Exception as e:
                print(e)
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
            top_p=0,
            stop=["\nObservation: "],
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
       
        action = match.group(1).strip().lower()
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
    st.title("Nuggt.io")
    guide = ""
    information = ""
    promotion = ""
    col1, col2 = st.columns([2, 3], gap="large")
    
    with st.sidebar:
        selected = option_menu(
            menu_title = "Choose an app",
            options = ["Nuggt Playground", "Search", "Data", "Document"],
        )
    
    if selected == "Nuggt Playground":
        guide = "This is where the guide will go."
        information = "This is where the information will go."
        promotion = "This is where the promotional text will go."

    if selected == "Search":
        guide = """Ask a question and our application would provide you answers with the relevant sources."""
        information = """This app was created with a single AI command as follows:
        ```Find websites related to {text:query} using {tool:google} and list the relevant sources```
        """
        promotion = """Nuggt allows you to create deployable AI apps with a single command. 
        You can create an app for anything you imagine. For example, the following command would create an app that researches
        on a given topic and creates a powerpoint presentation with the relevant sources:
        `Gather information on {text:topic} using {tool:google} and based on the gathered information create a powerpoint presentation with the relevant sources using {tool:powerpoint}` 
        You can create your own app at [Nuggt Playground](https://nuggt.io)
        """
        user_input = """Find websites related to {text:query} using {tool:google}. Browse three results to gather information on {text:query} using {tool:browse_website}. Display the results in the format <Content: Content of the website, URL: URL of the website> using {tool:display}"""

       # user_input = """Step: Research on {text:input} using {tool:google}
       # Step: From the results, browse 5 websites to get more information using {tool:browse_website}"""
        output_format = "I have displayed the results in the given format."

    if selected == "Data":
        guide = """Upload an Excel/CSV file for data analysis. You can ask the app generic questions like "summarise this data" or specific questions like "What is the relationship between Column A and Column B". The app can also generate data visualisations like bargraphs, piecharts etc. You can generate visualisations by typing "Generate visualisations for.." """
        information = """This app was created with a single AI command as follows:
        ```Open {upload:file} using {tool:document_loader} and complete the task {text:task} using {tool:data_analysis}```
        """
        promotion = """Nuggt allows you to create deployable AI apps with a single command. 
        You can create an app for anything you imagine. For example, the following command would create an app that fetches real-time data on stocks, saves it in an excel file and creates some basic visualisations for analysis.
        ```Find information on {text:stock} using {tool:stock_tool}, save it in an excel file using {tool:excel_tool} and display some basic visualisations using {tool:data_analysis}``` 
        You can create your own app at [Nuggt Playground](https://nuggt.io)
        """
        user_input = """Step: Open {upload:file} using {tool:python}
        Step: Display its head using {tool:python}
        Step: Complete the task {text:input} using {tool:python}
        Step: Display your results using {tool:display}
        """
        output_format = "I have displayed the visualisations using streamlit"

    if selected == "Document":
        guide = """Upload a PDF to retrieve information. Once you upload the PDF, you can simply type a question in the textbox to retrieve information and the relevant page number from the PDF. """
        
        information = """This app was created with a single AI command as follows:
        ```Open {upload:file} using {tool:document_loader} and answer {text:query} with sources using {tool:document_chat}```
        """

        promotion = """Nuggt allows you to create deployable AI apps with a single command. 
        You can create an app for anything you imagine. For example, the following command would create an app that retrieves data from a PDF and creates an excel sheet for it.
        ```Open {upload:file} using {tool:document_loader}, fetch data for {text:query} and organise it in an excel file using {tool:excel_tool}``` 
        You can create your own app at [Nuggt Playground](https://nuggt.io)
        """

        user_input = """Get past 7 days OHLC data for stock {text:stock} using {tool:python}. Calculate the MACD using {tool:python} (Do not use talib). Display the results in a table using {tool:display}. Display basic visualisations using {tool:display}."""

        output_format = "I have displayed the tables and visualisations for the given stock using streamlit."

    if selected in ["Nuggt Playground", "Search", "Data", "Document"]:
        with col1:
                st.subheader("How To Use")
                st.markdown(guide)
                st.subheader("How Was This Created?")
                st.markdown(information)
                st.subheader("Create Your Own App at [Nuggt Playground](https://nuggt.io)")
                st.markdown(promotion)
                
        with col2:
            if selected == "Nuggt Playground":
                user_input = st.text_area("Enter your instruction: ", key="enter_instruction")
                output_format = st.text_input("Enter output format: ", key="output")

            if user_input and output_format:
                save_to_sheets(user_input, output_format, "-", "-")
                variables = extract_variables(user_input)
                nuggt(user_input, output_format, variables)
    else:
        with st.form(key='my_form'):
            st.write('Please enter your information:')
            
            # Full Name
            full_name = st.text_input(label='Full Name')

            # Email
            email = st.text_input(label='Email')
            
            # Gender
            gender = st.selectbox('Gender', options=['Male', 'Female'])

            # Date of Birth
            dob = st.date_input('Date of Birth')

            # How do you plan to use nuggt?
            use_plan = st.text_area('How do you plan to use nuggt?')

            # Do you know how to get an OpenAI API Key?
            openai_key_knowledge = st.radio("Do you know how to get an OpenAI API Key?", ('Yes', 'No'))

            # Submit button
            submit_button = st.form_submit_button(label='Submit')

        # Validate email
        if submit_button:
            email_match = re.match(r"[^@]+@[^@]+\.[^@]+", email)
            if email_match is None:
                st.error('Invalid email address')
            else:
                st.success('Form successfully submitted.')
      
if __name__ == "__main__":
    main()














