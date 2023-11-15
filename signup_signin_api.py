from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, EmailStr, constr
from pymongo import MongoClient
from bson import ObjectId
from jose import JWTError, jwt
import uvicorn
import yaml

with open("path.yml", "r") as p:
    config = yaml.safe_load(p)
app = FastAPI()

# Connect to your MongoDB database
client = MongoClient(config['MONGODB_ATLAS_CLUSTER_URI'])
signup_collection = client[config["DB_NAME"]][config["SIGNUP_COLLECTION"]]
ALGORITHM = "HS256"

# Define a Pydantic model for the user data
class Signup_User(BaseModel):
    name: constr(min_length=1, max_length=50)
    email: EmailStr
    password: constr(min_length=6, max_length=50)

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
    not_required_paths = ["/signin", "/docs", "/openapi.json"]
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
