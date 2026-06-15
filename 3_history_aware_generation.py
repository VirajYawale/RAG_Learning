
import os

from dotenv import load_dotenv

from langchain_chroma import Chroma

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_groq import ChatGroq 

from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# connect to your document database
presistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(embedding_function=embedding_model, persist_directory=presistent_directory, collection_metadata={"hnsw:space": "cosine"})


# setup AI model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key = os.getenv("GROQ_API_KEY")
)

chat_history = []

def ask_question(user_question):
    print(f"--- You asked : ({user_question}) ---")

    # setup 1 : Make the question clear using conversation history
    if chat_history:
        # ask AI to make the question clear using the conversation history  
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip() # result.content.strip() is used to remove any leading or trailing whitespace from the rewritten question, ensuring that the search query is clean and formatted correctly for retrieval.

        print(f"Searching for: {search_question}")

    else:
        search_question = user_question
        print(f"Searching for: {search_question}")

    # setup 2 : Retrieve relevant documents from the vector store
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    # setup 3 : Generate answer using the retrieved documents and the original question
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Welcome to the AI assistant! Type 'exit' to quit.")
    while True:
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break
        ask_question(user_question)


if __name__ == "__main__":
    start_chat()