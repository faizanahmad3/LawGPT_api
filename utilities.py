import os
import openai
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import ConversationBufferMemory
from langchain.memory import MongoDBChatMessageHistory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


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
    docs_list = []
    for i in pdfs:
        full_path = os.path.join(pdfs_path, i)
        loader = PyPDFLoader(full_path)
        docs_list.extend(loader.load())
    docs = split_text(docs_list, 2500, 400)
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


def similarity_search_and_retriever(vectorstore, question):
    similar_docs = vectorstore.similarity_search(question, k=1)
    return vectorstore.as_retriever(search_kwargs={'filter': {'source': similar_docs[0].metadata['source']}})

def generating_response(question, template, retriever, config, session_id):
    message_history = MongoDBChatMessageHistory(connection_string=config['MONGODB_ATLAS_CLUSTER_URI'],
                                                database_name=config["DB_NAME"],
                                                collection_name=config["CHAT_HISTORY_COLLECTION"],
                                                session_id=session_id)

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
    print(result)
    message_history.add_user_message(question)
    message_history.add_ai_message(result['answer'])
    return result
