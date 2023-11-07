from fastapi import FastAPI, HTTPException, Depends, status, Security, Request
from pydantic import BaseModel, EmailStr, constr
from pymongo import MongoClient
from bson import ObjectId
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.hash import bcrypt
import uvicorn
import yaml

with open("path.yml", "r") as p:
    config = yaml.safe_load(p)
app = FastAPI()


# Define a Pydantic model for the user data
class Signup_User(BaseModel):
    name: constr(min_length=1, max_length=50)
    email: EmailStr
    password: constr(min_length=6, max_length=50)


class Signin_User(BaseModel):
    email: EmailStr
    password: constr(min_length=6, max_length=50)

# Connect to your MongoDB database
client = MongoClient(config['MONGODB_ATLAS_CLUSTER_URI'])
Mongodb_collection = client[config["DB_NAME"]][config["SIGNUP_COLLECTION"]]
ALGORITHM = "HS256"

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Exclude paths that don't require authentication
    not_required_paths = ["/signin/"]
    if request.url.path not in not_required_paths:  # Add any other paths that don't require authentication
        # Get the authorization header
        authorization: str = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="Not authenticated")

        # Extract the token from the header
        try:
            scheme, token = authorization.split()
            if scheme.lower() != 'bearer':
                raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        # Decode the token
        try:
            payload = jwt.decode(token, "secret", algorithms=ALGORITHM)
            user: str = payload.get("sub")
            if user is None:
                raise HTTPException(status_code=401, detail="Invalid token payload")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

        # You can perform additional checks here, like checking if the user exists, etc.

    # Pass the request to the next operation (whether another middleware or route)
    response = await call_next(request)
    return response

# Create a route for user registration
@app.post("/signup/")
async def signup(user: Signup_User):
    existing_user = Mongodb_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this email already exists")

    else:
        new_user = {
            "_id": ObjectId(),
            "name": user.name,
            "email": user.email,
            "password": user.password
        }
        Mongodb_collection.insert_one(new_user)

    return {"message": "User registered successfully", "user_id": str(new_user["_id"])}

# Generate an authentication token
def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, "secret", algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/signin/")
async def signin(user: Signin_User):
    existing_user = Mongodb_collection.find_one({"email": user.email})
    print(existing_user)
    if existing_user is None:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    access_token = create_access_token(data={"email": existing_user["email"], "id": str(existing_user["_id"])})
    decoder_jwt = jwt.decode(access_token, "secret", algorithms=ALGORITHM)
    print(access_token)
    return {"access_token": access_token, "decoder": decoder_jwt}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
