import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", # If you want to check the version, please go to the check_model.py
    temperature=0.7
)

response = llm.invoke("Explain langchain and its components")
print(response.content)
