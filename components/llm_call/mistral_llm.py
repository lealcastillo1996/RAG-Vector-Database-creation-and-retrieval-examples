
from dotenv import load_dotenv
# Load the environment variables from .env file
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI
chat = ChatMistralAI()

# Function to call MistralAI
def call_llm(prompt):
    messages = [HumanMessage(content=prompt)]
    response = chat.invoke(messages)
    return response
    







