from LawGPT_api import get_embeddings, generating_response, similarity_search_and_retriever, create_embeddings
import os
import yaml
import uuid
import openai
import uvicorn
from fastapi import FastAPI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

OpenAIEmbeddings.openai_api_key = os.getenv("OPENAI_API_KEY")
OpenAI.openai_api_key = os.getenv("OPENAI_API_KEY")

with open("path.yml", "r") as p:
    config = yaml.safe_load(p)

template = """
    Use the following pieces of context to answer the question at the end.
    I gave you a question, you have to understand the question, so think and then answer it.
    If you did not find any thing which is in the context, then print there is nothing about this question
    don't try to make up an answer.
    {chat_history}
    and the question is down below.
    question: {question}
    """
client = MongoClient(config['MONGODB_ATLAS_CLUSTER_URI'])
Mongodb_collection = client[config["DB_NAME"]][config["COLLECTION_NAME"]]
# create_embeddings(OpenAIEmbeddings(), config['pdfs_path'],config['ATLAS_VECTOR_SEARCH_INDEX_NAME'], Mongodb_collection)
embeddings = get_embeddings(OpenAIEmbeddings(), config['MONGODB_ATLAS_CLUSTER_URI'],config["DB_NAME"] ,config['COLLECTION_NAME'],config['ATLAS_VECTOR_SEARCH_INDEX_NAME'])

# question = "tell me about ramzan?"
# retriever = similarity_search_and_retriever(embeddings, question)
# answer = generating_response(question, template, retriever, config, str(uuid.uuid4()))

app = FastAPI(description="chatbot")
@app.post("/process_data")
async def process_data(question):
    retriever = similarity_search_and_retriever(embeddings, question)
    answer = generating_response(question, template, retriever, config, str(uuid.uuid4()))
    return answer

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)