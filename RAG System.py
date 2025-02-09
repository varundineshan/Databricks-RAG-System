# Databricks notebook source
# DBTITLE 1,Installing Necessary Modules
# MAGIC %pip install sentence-transformers faiss-cpu openai langchain python-docx

# COMMAND ----------

# DBTITLE 1,Reading Documents From File
from docx import Document
def read_docx(file_path):
    """Extract text from a .docx file."""
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Set the folder path where your .docx files are stored
folder_path = "/Workspace/Users/varlock4444@gmail.com/Documents"
documents = []   # This list will hold the text from each document
doc_names = []   # Optional: to track file names
files = os.listdir(folder_path)
print("Files in the foblder:", files)
for filename in os.listdir(folder_path):
    if filename.endswith(".docx"):
        file_path = os.path.join(folder_path, filename)
        text = read_docx(file_path)
        documents.append(text)
        doc_names.append(filename)

print(f"Loaded {len(documents)} documents.")

# COMMAND ----------

# DBTITLE 1,Calling Open AI ADA Embedding Model
from openai import AzureOpenAI
import numpy as np

endpoint="https://XXXX.openai.azure.com/"
subscription_key="XXXX" 

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2023-05-15",
)

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate an embedding for the given text using OpenAI's ada-002 model."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
   
    embedding = response.data[0].embedding
    return embedding

# Generate embeddings for all documents
document_embeddings = []
for doc in documents:
    emb = get_embedding(doc)
    document_embeddings.append(emb)

# Convert to a NumPy array with dtype float32 (required by FAISS)
document_embeddings = np.array(document_embeddings).astype("float32")

# COMMAND ----------

# DBTITLE 1,Indexing the Embeddings Using FAISS
# Indexing the Embeddings Using FAISS
import faiss
# Determine the dimensionality from one of the embeddings
embedding_dimension = document_embeddings.shape[1]
# Create a FAISS index (using L2 distance)
index = faiss.IndexFlatL2(embedding_dimension)
# Add the document embeddings to the index
index.add(document_embeddings)
print(f"Number of documents indexed: {index.ntotal}")

# COMMAND ----------

# DBTITLE 1,Building the Retrieval Pipeline
#Building the Retrieval Pipeline
import numpy as np
def retrieve_documents(query,k):
    """Retrieve the top-k relevant documents for the given query."""
    # Embed the query text using the same ada-002 model
    query_embedding = np.array([get_embedding(query)]).astype("float32")
    
    # Search the FAISS index for the nearest neighbors
    distances, indices = index.search(query_embedding, k)
    # For each retrieved index, create a dictionary with both document text and its name
    retrieved_docs = [
        {"doc_text": documents[i], "doc_name": doc_names[i]} 
        for i in indices[0]
    ]
    return retrieved_docs

# COMMAND ----------

# DBTITLE 1,Generating an Answer with a Generative Model
#Generating an Answer with a Generative Model

def generate_answer(query, context_docs):

    client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-08-01-preview"
)

    """Generate an answer using the retrieved context and the user query."""
    # Combine the context documents into a single string with document names as reference.
    # For each retrieved document, include a header with the document name followed by its text.
    context_str = "\n\n".join(
        [f"Document: {doc['doc_name']}\n{doc['doc_text']}" for doc in context_docs]
    )
    
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a knowledgeable assistant. Based on the following context, answer the question. Context: " + context_str
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                }
            ]
        }
    ]

    
    messages = chat_prompt
    # Generate the completion
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        top_p=0.95
    )
    
    completion_choices = completion.__getattribute__('choices')
    # Access the first choice and its message
    first_choice = completion_choices[0]
    answer = first_choice.message.content 
    return answer


# COMMAND ----------

# DBTITLE 1,End User Query And Function Calls

user_query="how much will it cost to create rag model with 30 page document also mention what technologies are included in the SRS document"
# Retrieve the top relevant documents
retrieved_context = retrieve_documents(user_query, k=len(documents))
# Generate an answer using the retrieved context
answer = generate_answer(user_query, retrieved_context)
print(answer)
doc_name=[]
print("\nReferenced Documents:")
for doc in retrieved_context:
    doc_name.append(doc["doc_name"])
print(doc_name)