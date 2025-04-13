import chainlit as cl
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Define the prompt template for the chatbot
prompt_template = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="""
    You are a helpful assistant. Use the conversation history and respond to the user's query:
    Conversation History: {chat_history}
    User Input: {user_input}
    """
)

# Initialize memory for conversation context
conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the LLM (OpenAI GPT model)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

# Create an LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template, memory=conversation_memory)

@cl.on_chat_start
def start_chat():
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def handle_message(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    response = await llm_chain.acall({"user_input": message.content})
    await cl.Message(content=response["text"]).send()
