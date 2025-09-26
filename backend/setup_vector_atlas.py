# setup_vectordb_atlas.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

def main():
    """
    Main function to load data, split it into chunks, generate embeddings,
    and store them in a MongoDB Atlas Vector Search index.
    """
    # Load environment variables
    load_dotenv()
    google_api_key = os.getenv('MONGO_CONNECTION_STRING')
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in the .env file.")
    
    # It's better to load the Atlas connection string from the .env file too
    mongo_conn_string = os.getenv('MONGO_CONNECTION_STRING')
    if not mongo_conn_string:
        raise ValueError("MONGO_CONNECTION_STRING not found in the .env file.")

    # Define your database and collection names
    DB_NAME = "coffee_db"
    COLLECTION_NAME = "knowledge_collection"
    
    print("Processing the knowledge base for MongoDB Atlas...")

    # 1. Load the document
    loader = TextLoader('./coffee_knowledge.txt', encoding='utf-8')
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} document(s).")

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split the document into {len(chunks)} chunks.")

    # 3. Generate embeddings and store them in MongoDB Atlas
    print("Generating embeddings with Google's model and uploading to Atlas...")
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Connect to your MongoDB cluster
    client = MongoClient(mongo_conn_string)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    # The main call to upload the documents and their embeddings
    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection=collection,
        index_name="default" # The name you gave your vector search index in the Atlas UI
    )
    
    print("--------------------------------------------------")
    print("Vector data has been successfully uploaded to MongoDB Atlas!")
    print(f"Database: {DB_NAME}, Collection: {COLLECTION_NAME}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()