import socket
import threading
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
import os
import requests

server_type = os.getenv('SERVER_TYPE', 'api')  # Set this environment variable to "api" or "local"

def make_api_call(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 500,
        'do_sample': True,
        'temperature': 0.01,
        'top_p': 0.01,
        'repetition_penalty': 1,
        'top_k': 50,
        'stopping_strings': ["Observation: "]
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['text']
        return result

def generate(message):
    prompt_template = """

    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {query}

    ### Response:"""
    prompt = prompt_template.format(query=message)
    if server_type == 'local':
        outputs = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.01, top_k=50, top_p=0.01, eos_token_id=28583, repetition_penalty=1)
        return outputs[0]["generated_text"].split("### Response:")[1]
    else:
        result = make_api_call(prompt)
        return result

def handle_client(client_socket):
    global first
    while True:
        # Receive data from the client
        data = ""
        while "<|end|>" not in data: 
            chunk = client_socket.recv(1024).decode()
            data = data + chunk
    
        if not data:
            break
        
        data = data.replace("<|end|>", "")
        print('Received from client: {}'.format(data))
        
        # Process the data and generate a response
        response = generate(data.strip() + "\nWrite the next Step/Action/Action Input/Observation.")
        
        response = response + " <ENDMESSAGE>"
        
        # Send the response back to the client
        client_socket.send(response.encode())
    
    # Close the client socket
    client_socket.close()

if __name__ == "__main__":
    # Define the server host and port
    host = '0.0.0.0'  # Listen on all network interfaces
    port = 8082
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the host and port
    server_socket.bind((host, port))
    
    # Listen for incoming connections
    server_socket.listen(1)
    
    print('Server listening on {}:{}'.format(host, port))
    
    if server_type == 'local':
        print("Loading Model...")
        model_name_or_path = "TheBloke/WizardCoder-15B-1.0-GPTQ"
        # Or to load it locally, pass the local download path
        # model_name_or_path = "/path/to/models/TheBloke_WizardCoder-15B-1.0-GPTQ"
        
        use_triton = False
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        
        model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                use_safetensors=True,
                device="cuda:0",
                use_triton=use_triton,
                quantize_config=None)
        
        # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        logging.set_verbosity(logging.CRITICAL)
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("Model Loaded")
        
    elif server_type == 'api':
        # Replace with your Oobabooga API url. Only https is supported, no streaming.
        booga_api_url = "https://insertion-friends-trial-paperbacks.trycloudflare.com/api"
        URI = booga_api_url + "/v1/generate"
    else:
        raise ValueError('Invalid server type. Choose "local" or "api".')

    # Accept incoming client connections
    while True:
        client_socket, addr = server_socket.accept()
        print('Client connected: {}'.format(addr))
        
        # Start a new thread to handle the client connection
        client_handler_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler_thread.start()
