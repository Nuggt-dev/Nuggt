import os
import sys
import traceback
from io import StringIO

import openai
from colorama import Fore


class PythonREPL:
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

        return (
            output
            if not error
            else "Your code has the following error. Please provide the corrected code.\n"
            + error
        )


python_repl = PythonREPL()


def extract_code_from_block(text):
    if "!pip" in text:
        return "The package is successfully installed."
    text = text.replace("`", "")
    text = text.replace("python", "")
    return text


def fix_error(code, result):
    while "Your code has the following error." in result:
        error = result.replace(
            "Your code has the following error. Please provide the corrected code.", ""
        )
        user_input = f"Input:\n{code}\nError:\n{error}\nOutput:"

        print(f"I am going to correct: {user_input}")
        print(Fore.RED + "Code needs some correction.")
        messages = [
            {
                "role": "user",
                "content": f"""Output the corrected code in the following format:\n```Your code here```\n{user_input}""",
            },
        ]

        response = openai.ChatCompletion.create(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("MODEL_NAME"),
            messages=messages,
            temperature=0,
        )

        code = response.choices[0].message["content"].strip()
        code = extract_code_from_block(code)
        print(f"\nOutput of the model: {code}")

        result = python_repl.run(code)

    print(Fore.GREEN + "Code has been corrected.")
    return result


def python(code):
    code = extract_code_from_block(code)
    if code == "The package is successfully installed.":
        return code
    result = python_repl.run(code)
    if "Your code has the following error." in result:
        result = fix_error(code, result)

    if result == "":
        result = "Your code was successfully executed."

    return result
