import os
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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

def get_embeddings(embeddings, Mongodb_Atlas_Cluster_URI,DB_Name, Collection_Name,Atlas_Vector_Search_Index_Name):
    return MongoDBAtlasVectorSearch.from_connection_string(
        Mongodb_Atlas_Cluster_URI,
        DB_Name + "." + Collection_Name,
        embedding=embeddings,
        index_name=Atlas_Vector_Search_Index_Name)
def similarity_search(vectorstore, question, collection_db):
    similar_docs = vectorstore.similarity_search(question, k=1)
    multiple_dict_metadata = collection_db.find({'source': similar_docs[0].metadata['source']}, {"source": 1})
    return [doc for doc in multiple_dict_metadata]

def retrieve_data(question, template, vectorstore, metadata):
    QA_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    retriever = vectorstore.as_retriever(metadata=metadata, return_metadata = True)
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_prompt},
        verbose = True
    )
    return qa.run(question)
