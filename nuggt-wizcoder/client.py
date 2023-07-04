import socket

# Define the server host and port
def generate(message):
    host = '127.0.0.1' #localhost
    port = 8081

    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((host, port))

    # Send a message to the server
    client_socket.send(message.encode())

    # Receive the response from the server
    response = ""
    while "<ENDMESSAGE>" not in response: 
        print("Coming in...")
        chunk = client_socket.recv(1024).decode()
        response = response + chunk
    
    response = response.replace("<ENDMESSAGE>", "")
    #response = client_socket.recv(1024).decode()

    # Close the client socket
    client_socket.close()
    return response

