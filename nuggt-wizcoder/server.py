import socket
import threading
import argparse
import os
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import requests

# Define the server host and port
host = '0.0.0.0'  # Listen on all network interfaces
port = 8081

# Set SERVER_TYPE env variable to local or api
# For local inference, use: export SERVER_TYPE=local
# For API inference, use: export SERVER_TYPE=api

# model_name_or_path is only used for local inference. For API inference, it is ignored.
model_name_or_path = "TheBloke/WizardCoder-15B-1.0-GPTQ"

# Adjust the API url parameter here. This is only used for API inference.
# Oobabooga API url. Only https is supported, no streaming.
booga_api_url = "https://sql-minute-found-blend.trycloudflare.com/api"
URI = booga_api_url + "/generate"


class BaseServer:

    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print('Server listening on {}:{}'.format(host, port))
        print("Loading Model...")

    def generate(self, message):
        raise NotImplementedError

    def handle_client(self, client_socket):
        while True:
            data = ""
            while "" not in data:
                chunk = client_socket.recv(1024).decode()
                data = data + chunk

            if not data:
                break

            data = data.replace("", "")
            print('Received from client: {}'.format(data))

            # Process the data and generate a response
            response = self.generate(data.strip() + "\nWrite the next Step/Action/Action Input/Observation.")
            response = response + " <ENDMESSAGE>"

            # Send the response back to the client
            client_socket.send(response.encode())

        # Close the client socket
        client_socket.close()

    def run(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print('Client connected: {}'.format(addr))

            # Start a new thread to handle the client connection
            client_handler_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_handler_thread.start()


class LocalInferenceServer(BaseServer):

    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                                                        use_safetensors=True,
                                                        device="cuda:0",
                                                        use_triton=False,
                                                        quantize_config=None)

        logging.set_verbosity(logging.CRITICAL)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        print("Model Loaded")

    def generate(self, message):
        prompt_template = """
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {query}

        ### Response:"""

        prompt = prompt_template.format(query=message)
        outputs = self.pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.01, top_k=50,
                            top_p=0.01, eos_token_id=28583, repetition_penalty=1)
        return outputs[0]["generated_text"].split("### Response:")[1]


class APIInferenceServer(BaseServer):
    def __init__(self):
        super().__init__()

    def make_api_call(self, prompt):
        request = {
            'prompt': prompt,
            'max_new_tokens': 500,
            # Feel free to change the parameters below, but from my experiment are the best for code generation
            'do_sample': False,
            'temperature': 0.2,
            'top_p': 0.95,
            'typical_p': 1,
            'epsilon_cutoff': 0,  # In units of 1e-4
            'eta_cutoff': 0,  # In units of 1e-4
            'tfs': 1,
            'top_a': 0,
            'repetition_penalty': 1,
            'repetition_penalty_range': 0,
            'top_k': 50,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,

            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True,
            'stopping_strings': []
        }

        response = requests.post(URI, json=request)

        if response.status_code == 200:
            result = response.json()['results'][0]['text']
            return result

    def generate(self, message):
        prompt_template = """
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {query}

        ### Response:"""

        prompt = prompt_template.format(query=message)
        result = self.make_api_call(prompt)
        return result.split("### Response:")[1]


if __name__ == "__main__":
    server_type = os.getenv('SERVER_TYPE', 'local')  # Set this environment variable to "api" or "local"
    if server_type == 'local':
        server = LocalInferenceServer()
    elif server_type == 'api':
        server = APIInferenceServer()
    else:
        raise ValueError('Invalid server type. Choose "local" or "api".')

    server.run()
