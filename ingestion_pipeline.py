
import os # For environment variable management

import sentence_transformers 

from langchain_community.document_loaders import TextLoader, DirectoryLoader # For loading text documents

from langchain_text_splitters import CharacterTextSplitter # type: ignore # For splitting text into manageable chunks

# from langchain_openai import OpenAIEmbeddings # For generating embeddings using OpenAI's API 

from langchain_community.embeddings import HuggingFaceEmbeddings # don't have credits in openai api key, so using this huggingface embedding

from langchain_chroma import Chroma # For storing and querying embeddings in a vector database

from dotenv import load_dotenv 

load_dotenv()



# loading function

def load_documents(docs_path = "docs"):

    print(f"Loading documents from {docs_path}...")

    # Check if the specified directory exists
    if not os.path.exists(docs_path):
        print(f"Directory {docs_path} does not exist. Please check the path and try again.")

    # load all .txt files in the specified directory
    loader = DirectoryLoader(docs_path, glob="**/*.txt",loader_cls=TextLoader) # here we use TextLoader, it will be depending on the type of documents you have, you can use other loaders like PDFLoader, DocxLoader, etc.

    documents = loader.load()

    if(len(documents) == 0):
        print(f"No documents found in {docs_path}. Please add your company document.")
        
    # for i, doc in enumerate(documents[:2]): # Print the source and a snippet of the content for the first 2 documents
    #     print(f"Document {i+1}: {doc.metadata['source']}" )
    #     print(f"Content: {doc.page_content[:50]}...")
    #     print("----------------------------------------")
    return documents




# chunking function

def split_documents(documents, chunk_size=800, chunk_overlap=0): # chunk_size is the maximum number of characters in each chunk, and chunk_overlap is the number of characters that overlap between consecutive chunks. Adjust these parameters based on your needs and the nature of your documents.

    print(f"Splitting documents into chunks of size {chunk_size} with overlap of {chunk_overlap}...")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # split_document is most basic spliting technique in langchain, it will split the document into chunks based on the specified chunk size and overlap. You can also use other splitting techniques like RecursiveCharacterTextSplitter, TokenTextSplitter, etc. depending on your needs.
    chunks = text_splitter.split_documents(documents)

    # if chunks:
        
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"Chunk {i+1}: {chunk.metadata['source']} - {chunk.page_content[:50]}...")
    #         print("-" * 50)

    #     if len(chunks) > 5:
    #         print(f"... and {len(chunks) - 5} more chunks.")

    return chunks





# embedding and vector store function

def create_vector_store(chunks, persist_directory="db/chroma_db"):

    print(f"Create embeddings and vector store in ChromaDB, persisting to {persist_directory}...")

    # Initialize OpenAI embeddings with your API key )
    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")) # here we are using text-embedding-3-small model, you can choose other models based on your needs and the size of your documents. Make sure to check OpenAI's documentation for available embedding models and their capabilities.

    # don't have OpenAI API credits, so i will be using model_name="sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a Chroma vector store and persist it to disk
    print("Creating vector store and persisting to disk...")
    vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"}) #collection_metadata is used to specify the distance metric for the vector store. In this case, we are using cosine similarity, which is a common choice for text embeddings.

    print("Vector store created and persisted successfully.")

    return vector_store







def main():
    print("Starting ingestion pipeline...")

    # Step 1: Load documents
    documents = load_documents(docs_path="docs")

    # Step 2: Split documents into chunks
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=0)

    # Step 3: Create vector store and persist it to disk
    vector_store = create_vector_store(chunks, persist_directory="db/chroma_db")


if __name__ == "__main__":
    main()

