import json
import logging
from typing import List
from datetime import datetime
from pydantic import BaseModel, EmailStr, constr, validator
import re

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict

logger = logging.getLogger(__name__)

DEFAULT_DBNAME = "chat_history"
DEFAULT_COLLECTION_NAME = "message_store"


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in MongoDB.

    Args:
        connection_string: connection string to connect to MongoDB
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        database_name: name of the database to use
        collection_name: name of the collection to use
    """

    def __init__(
            self,
            connection_string: str,
            session_id: str,
            user_id: str,
            database_name: str = DEFAULT_DBNAME,
            collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        from pymongo import MongoClient, errors

        self.connection_string = connection_string
        self.session_id = session_id
        self.user_id = user_id
        self.database_name = database_name
        self.collection_name = collection_name

        try:
            self.client: MongoClient = MongoClient(connection_string)
        except errors.ConnectionFailure as error:
            logger.error(error)

        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.collection.create_index("SessionId")
        self.collection.create_index("user_id")

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from MongoDB"""
        cursor = None
        from pymongo import errors

        try:
            cursor = self.collection.find({"SessionId": self.session_id})
        except errors.OperationFailure as error:
            logger.error(error)

        if cursor:
            items = [json.loads(document["History"]) for document in cursor]
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in MongoDB"""
        from pymongo import errors

        try:
            self.collection.insert_one(
                {
                    "UserId": self.user_id,
                    "SessionId": self.session_id,
                    "History": json.dumps(_message_to_dict(message)),
                    "Date_Time": str(datetime.now())
                }
            )
        except errors.WriteError as err:
            logger.error(err)

    def clear(self) -> None:
        """Clear session memory from MongoDB"""
        from pymongo import errors

        try:
            self.collection.delete_many({"SessionId": self.session_id})
        except errors.WriteError as err:
            logger.error(err)

# Define a Pydantic model for the user data
class Signup_User(BaseModel):
    name: constr(min_length=1, max_length=30)
    email: EmailStr
    password: constr(min_length=6, max_length=30)

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
    password: constr(min_length=6, max_length=30)

class for_question(BaseModel):
    question: str
    userid: str

class for_userid(BaseModel):
    userid: str

class for_user_session(BaseModel):
    userid: str
    sessionid: str