from utilities import get_embeddings, generating_response, similarity_search_and_retriever, create_embeddings, get_sessionid, get_history
import os
import yaml
import uuid
from bson import ObjectId
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, constr, validator
import re
import openai
import uvicorn
from fastapi import FastAPI, HTTPException, Request
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
embeddings_collection = client[config["DB_NAME"]][config["EMBEDDINGS_COLLECTION"]]
chat_history_collection = client[config["DB_NAME"]][config["CHAT_HISTORY_COLLECTION"]]
signup_collection = client[config["DB_NAME"]][config["SIGNUP_COLLECTION"]]

# create_embeddings(OpenAIEmbeddings(), config['pdfs_path'], config['ATLAS_VECTOR_SEARCH_INDEX_NAME'], embeddings_collection)
embeddings = get_embeddings(OpenAIEmbeddings(), config['MONGODB_ATLAS_CLUSTER_URI'],config["DB_NAME"] ,config['EMBEDDINGS_COLLECTION'],config['ATLAS_VECTOR_SEARCH_INDEX_NAME'])

ALGORITHM = "HS256"
app = FastAPI(title="Law_GPT API" ,description="ChatBot")
session_id = None
@app.post("/search")
async def search(question):
    global session_id
    if session_id is None:
        session_id = str(uuid.uuid4())
    retriever = similarity_search_and_retriever(embeddings, question)
    answer = generating_response(question, template, retriever, config, session_id)
    return answer

@app.get("/SessionId")
async def sessionid():
    return get_sessionid(chat_history_collection)
@app.get("/history")
async def history(SessionId):
    return get_history(SessionId, chat_history_collection)

# Define a Pydantic model for the user data
class Signup_User(BaseModel):
    name: constr(min_length=1, max_length=50)
    email: EmailStr
    password: constr(min_length=6, max_length=50)
    @validator('password')
    def password_complexity(cls, value):
        # if not re.search("[!@#$%^&*(),.?\":{}|<>]", value):
        #     raise ValueError('Password must contain at least one special character')
        # if not re.search("[a-z]", value):
        #     raise ValueError('Password must contain at least one lowercase letter')
        # if not re.search("[A-Z]", value):
        #     raise ValueError('Password must contain at least one uppercase letter')
        if not re.search("[0-9]", value):
            raise ValueError('Password must contain at least one digit')
        return value

class Signin_User(BaseModel):
    email: EmailStr
    password: constr(min_length=6, max_length=50)

# Generate an authentication token
def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, "secret", algorithm=ALGORITHM)
    return encoded_jwt
# Create a route for user registration
@app.post("/signup")
async def signup(user: Signup_User):
    existing_user = signup_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this email already exists")

    else:
        new_user = {
            "_id": ObjectId(),
            "name": user.name,
            "email": user.email,
            "password": user.password
        }
        signup_collection.insert_one(new_user)
        return create_access_token(data={"email": new_user["email"], "id": str(new_user["_id"])})

@app.post("/signin")
async def signin(user: Signin_User):
    existing_user = signup_collection.find_one({"email": user.email})
    print(existing_user)
    if existing_user is None:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    return create_access_token(data={"email": existing_user["email"], "id": str(existing_user["_id"])})

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Exclude paths that don't require authentication
    not_required_paths = ["/signin", "/docs", "/openapi.json", "/SessionId", "/history", "/signup", "/search"]
    if request.url.path not in not_required_paths:  # Add any other paths that don't require authentication
        # Get the authorization header
        authorization: str = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="Not authenticated")

        try:
            payload = jwt.decode(authorization, "secret", algorithms=ALGORITHM)
            if payload is None:
                raise HTTPException(status_code=401, detail="Invalid token payload")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    # Pass the request to the next operation (whether another middleware or route)
    response = await call_next(request)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)