import os

from langchain_chroma import Chroma

import sentence_transformers 

from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

# I don't have API key credits
# from langchain_openai import ChatOpenAI # For generating answers using OpenAI's API, you can replace this with any other language model you prefer, such as HuggingFace's transformers or Cohere's API, depending on your requirements and available resources.

# so we will use Ollama with Llama3
# from langchain_ollama import ChatOllama

# or use groq
from langchain_groq import ChatGroq 


from langchain_core.messages import HumanMessage, SystemMessage # For structuring the input to the language model in a conversational format


load_dotenv()

presistent_directory = "db/chroma_db"

# load embeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(embedding_function=embedding_model, persist_directory=presistent_directory, collection_metadata={"hnsw:space": "cosine"})

#search for relevant documents
# query = "What was NVIDIA's first graphics accelerator called?"
query = "What is my name?"

# retrieval = db.as_retriever(search_kwargs={"k": 3}) # k is the number of relevant documents to retrieve
# we retrive beause we want to get the relevant documents from the vector store based on the query, and then we can use those documents to answer the question or provide more context.


retrieval = db.as_retriever(search_type = "similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 5}) 
# here we are using similarity_score_threshold search type, which allows us to set a threshold for the similarity score. Only documents with a similarity score above the threshold will be retrieved. This can help filter out less relevant documents and improve the quality of the results.

relevant_docs = retrieval.invoke(query)

print("-" * 50)

print(f"Query: {query}")

print("-" * 50)

# Display the retrieved documents and their similarity scores

# print("---Context---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}: \n {doc.page_content}")
#     print("-" * 50)








# Now the answer generation
# some dependacies are imported

# combine the retrieved documents into a single context string with query
combined_input = f"""Based on the following retrieved documents, answer the question: {query}\n\n

Documents: {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs]) }\n

Please provide a clear, helpful answer based on the information from the documents. If the answer is not found in the documents, please indicate that the information is not available."""



# create a language model instance
# model = ChatOpenAI(model = "gpt-3.5-turbo")  ---  it ask for API key, and i don't have enough credits, so i will be using HuggingFace's transformers instead.

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key = os.getenv("GROQ_API_KEY")
)


# define the messages for the language model
messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided context."),
    HumanMessage(content=combined_input)
]

# Invoke the model with the combined input
result = model.invoke(messages)

print("\n---Generated Response---")

print("Content only: ")
print(result.content)


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"

