import os
import yaml
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

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
    for i in pdfs:
        full_path = os.path.join(pdfs_path, i)
        loader = PyPDFLoader(full_path)
        pages = loader.load()
        docs = split_text(pages, 2500, 400)
        emb_dir = folder_path + i[:-4]
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

def retrieve_data(question, template, vectorstore):
    QA_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    retriever = vectorstore.as_retriever(search_type="similarity")
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                    chain_type="stuff",
                                    retriever=retriever,
                                    chain_type_kwargs={"prompt": QA_prompt},
                                    verbose=True)
    return qa.run(question)

if __name__ == "__main__":
    OpenAIEmbeddings.openai_api_key = os.getenv("OPENAI_API_KEY")
    OpenAI.openai_api_key = os.getenv("OPENAI_API_KEY")
    question = "tell me about murder case"
    template = """
    Use the following pieces of context to answer the question at the end.
    I gave you a question, you have to take out the keywords from that question and search on the basis of keywords similarity.
    If you did not find anything similar to the keywords, then print 'I didn't find any similarity.'
    Don't try to make up an answer.
    {context}
    question is down below.
    question: {question}
    """
    with open("path.yml", "r") as p:
        config = yaml.safe_load(p)

    openai_embedding = OpenAIEmbeddings()
    # create_embeddings(openai_embedding, config['pdfs_path'],config['embedding_path'])  # if u want to create embeddings then remove comment from this line

    embeddings = get_embeddings(openai_embedding, config['embedding_path'])
    answer = retrieve_data(question, template, embeddings)
    print(answer)
