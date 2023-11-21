import os
import json
from datetime import datetime
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import ConversationBufferMemory
from langchain.memory import MongoDBChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage


def split_text(pages, chunksize, chunkoverlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(pages)
    return docs


def create_embeddings(embeddings, pdfs_path, Atlas_Vector_Search_Index_Name, Mongodb_collection):
    pdfs = os.listdir(pdfs_path)
    for i in pdfs:
        full_path = os.path.join(pdfs_path, i)
        loader = PyPDFLoader(full_path)
        docs = split_text(list(loader.load()), 2500, 400)
        MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embeddings,
            collection=Mongodb_collection,
            index_name=Atlas_Vector_Search_Index_Name,
        )


def get_embeddings(embeddings, Mongodb_Atlas_Cluster_URI, DB_Name, Collection_Name, Atlas_Vector_Search_Index_Name):
    return MongoDBAtlasVectorSearch.from_connection_string(
        Mongodb_Atlas_Cluster_URI,
        DB_Name + "." + Collection_Name,
        embedding=embeddings,
        index_name=Atlas_Vector_Search_Index_Name)


def similarity_search(vectorstore, question):
    similar_docs = vectorstore.similarity_search(question, k=1)
    return similar_docs

def QA_retriever(vectorstore, similar_docs):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={'filter': {'source': similar_docs[0].metadata['source']}})

def generating_response(question, template, retriever, config, session_id):
    message_history = MongoDBChatMessageHistory(connection_string=config['MONGODB_ATLAS_CLUSTER_URI'],
                                                database_name=config["DB_NAME"],
                                                collection_name=config["CHAT_HISTORY_COLLECTION"],
                                                session_id=session_id,
                                                )
    QA_prompt = PromptTemplate(input_variables=["question", "chat_history"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(),
        retriever=retriever,
        memory=memory,
        condense_question_prompt=QA_prompt,
        verbose=True,
    )
    result = qa({"question": question, "chat_history": message_history.messages})
    message_history.add_message(message=BaseMessage(type="human", content=question, date_time=str(datetime.now())))
    message_history.add_message(message=BaseMessage(type="ai", content=result["answer"], date_time=str(datetime.now())))
    return result


def get_sessionid(collection):
    return json.dumps(collection.distinct('SessionId'))


def get_history(SessionId, collection):
    history = [x["History"] for x in list(collection.find({'SessionId': SessionId}, {'_id': 0, 'History': 1}))]
    chat_strings = []
    for single_doc in history:
        single_doc = json.loads(single_doc)
        if single_doc['type'] == "human":
            chat_strings.append({"human": str(single_doc["data"]["content"])})
        elif single_doc['type'] == "ai":
            chat_strings.append({"ai": str(single_doc["data"]["content"])})
    return json.dumps(chat_strings)
