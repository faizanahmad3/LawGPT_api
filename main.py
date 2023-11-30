from utilities import get_embeddings, generating_response, similarity_search, QA_retriever, create_embeddings, \
    get_sessionid, get_history, get_userid, create_access_token
import os
import yaml
import uuid
from bson import ObjectId
from jose import JWTError, jwt
import openai
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
from Classes import Signup_User, Signin_User, for_question, for_userid, for_user_session

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
    and the question is down below.
    question: {question}
    """
client = MongoClient(config['MONGODB_ATLAS_CLUSTER_URI'])
embeddings_collection = client[config["DB_NAME"]][config["EMBEDDINGS_COLLECTION"]]
chat_history_collection = client[config["DB_NAME"]][config["CHAT_HISTORY_COLLECTION"]]
signup_collection = client[config["DB_NAME"]][config["SIGNUP_COLLECTION"]]

# create_embeddings(OpenAIEmbeddings(), config['pdfs_path'], config['ATLAS_VECTOR_SEARCH_INDEX_NAME'], embeddings_collection)
embeddings = get_embeddings(OpenAIEmbeddings(), config['MONGODB_ATLAS_CLUSTER_URI'], config["DB_NAME"],
                            config['EMBEDDINGS_COLLECTION'], config['ATLAS_VECTOR_SEARCH_INDEX_NAME'])

ALGORITHM = "HS256"
app = FastAPI(title="Law_GPT API", description="ChatBot")
session_id = None
retriever = None
similarity = None

@app.post("/search")
async def search(query: for_question):
    global session_id, retriever, similarity
    print(query.question)
    if session_id is None:
        session_id = str(uuid.uuid4())
        similarity = similarity_search(embeddings, query.question)
    retriever = QA_retriever(embeddings, similarity)
    answer = generating_response(query, template, retriever, config, session_id)
    print(answer)
    return answer


@app.get("/userid")
async def userid(request: Request):
    authorization: str = request.headers.get("Authorization")
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = get_userid(authorization, ALGORITHM)
        return user_id
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/SessionId")
async def sessionid(user_id: for_userid):
    return get_sessionid(user_id, chat_history_collection)


@app.get("/history")
async def history(userid_sessionid: for_user_session):
    return get_history(userid_sessionid, chat_history_collection)


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
        return create_access_token(data={"email": new_user["email"], "id": str(new_user["_id"])}, ALGORITHM=ALGORITHM)


@app.post("/signin")
async def signin(user: Signin_User):
    existing_user = signup_collection.find_one({"email": user.email})
    print(existing_user)
    if existing_user is None:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    return create_access_token(data={"email": existing_user["email"], "id": str(existing_user["_id"])},
                               ALGORITHM=ALGORITHM)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Exclude paths that don't require authentication
    not_required_paths = ["/signin", "/docs", "/openapi.json", "/history", "/SessionId", "/signup", "/search"]
    if request.url.path not in not_required_paths:  # Add any other paths that don't require authentication
        # Get the authorization header
        authorization: str = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="Not authenticated")

        try:
            credentials = jwt.decode(authorization, "secret", algorithms=ALGORITHM)
            if credentials is None:
                raise HTTPException(status_code=401, detail="Invalid token credentials")

        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    # Pass the request to the next operation (whether another middleware or route)
    response = await call_next(request)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
