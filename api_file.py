from LawGPT_api import get_embeddings, retrieve_data, similarity_search, create_embeddings
import os
import yaml
import openai
import uvicorn
from fastapi import FastAPI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

OpenAIEmbeddings.openai_api_key = os.getenv("OPENAI_API_KEY")
OpenAI.openai_api_key = os.getenv("OPENAI_API_KEY")

with open("path.yml", "r") as p:
    config = yaml.safe_load(p)

template = """
#     Use the following pieces of context to answer the question at the end.
#     I gave you a question, you have to take out the keywords from that question and search on the basis of keywords similarity.
#     If you did not find anything similar to the keywords, then print 'I didn't find any similarity.'
#     Don't try to make up an answer.
#     {context}
#     question is down below.
#     question: {question}
#     """

# create_embeddings(OpenAIEmbeddings(), config['pdfs_path'],config['embedding_path'])
embeddings = get_embeddings(OpenAIEmbeddings(), config['embedding_path'])

app = FastAPI()
@app.post("/process_data")
async def process_data(question):
    metadata = similarity_search(embeddings, question)
    answer = retrieve_data(question, template, embeddings, metadata)
    return answer

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)