import os
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA


def split_text(pages, chunksize, chunkoverlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(pages)
    return docs

def create_embeddings(embeddings, pdfs_path, folder_path):
    pdfs = os.listdir(pdfs_path)
    docs_list = []
    for i in pdfs:
        full_path = os.path.join(pdfs_path, i)
        loader = PyPDFLoader(full_path)
        docs_list.extend(loader.load())
    docs = split_text(docs_list, 2500, 400)
    emb_dir = folder_path
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=emb_dir
    )

def get_embeddings(embeddings, folder_path):
    return Chroma(
        persist_directory=folder_path,
        embedding_function=embeddings
    )
def similarity_search(vectorstore, question):
    similar_docs = vectorstore.similarity_search(question, k=2)
    multiple_dict_metadata = []
    for doc in range(len(similar_docs)):
        multiple_dict_metadata.append(similar_docs[doc].metadata)
    return multiple_dict_metadata

def retrieve_data(question, template, vectorstore, metadata):
    QA_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    retriever = vectorstore.as_retriever(metadata=metadata, return_metadata = True)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                    chain_type="stuff",
                                    retriever=retriever,
                                    chain_type_kwargs={"prompt": QA_prompt},
                                    verbose=True)
    return qa.run(question)
