import os
import openai
import pinecone
from dotenv import load_dotenv
from pinecone.deployment import Model
from pinecone.service.action import ActionType
from pinecone.service.model import ModelType

# Load API keys
load_dotenv()
pinecone_api_key = os.getenv("810a704b-8786-4a9b-acce-90893a4b9a84")
openai_api_key = os.getenv("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")

# Initialize Pinecone and OpenAI
pinecone.init(api_key=pinecone_api_key)
openai.api_key = openai_api_key

# Create a new Pinecone namespace
pinecone.deinit()
pinecone.init(api_key=pinecone_api_key)
pinecone.create_namespace("chatbot_namespace")

# Define a custom Pinecone model
class ChatbotModel(Model):
    def __init__(self):
        super().__init__(
            model_type=ModelType.custom,
            action_type=ActionType.custom,
            input_type="str",
            output_type="str",
        )

    def call(self, query):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"{query}\n\nAI:",
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.8,
        )
        return response.choices[0].text.strip()

# Deploy the custom model
pinecone.deployment.create_model("chatbot_model", model_cls=ChatbotModel)
pinecone.deployment.deploy("chatbot_model", "chatbot_namespace")

# Initialize the chatbot
chatbot = pinecone.deployment.load_model("chatbot_model")

# Initialize the conversation history storage
pinecone.create_index(index_name="conversations", metric="cosine", shards=1)

# Start a new conversation
conversation_id = "conversation_1"
pinecone.upsert(items={conversation_id: []}, index_name="conversations")

# Chat with the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Fetch conversation history
    conversation_history = pinecone.fetch(ids=[conversation_id], index_name="conversations")[conversation_id]

    # Update conversation history with user input
    conversation_history.append(f"User: {user_input}")

    # Generate chatbot response using conversation history
    context = "\n".join(conversation_history)
    response = chatbot(context)

    # Update conversation history with chatbot response
    conversation_history.append(f"AI: {response}")

    # Store updated conversation history
    pinecone.upsert(items={conversation_id: conversation_history}, index_name="conversations")

    print(f"AI: {response}")

# Cleanup
pinecone.deployment.undeploy("chatbot_model", "chatbot_namespace")
pinecone.deployment.delete_model("chatbot_model")
pinecone.deinit()
pinecone.drop_index(index_name="conversations")
