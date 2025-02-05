import os
import requests
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  


# Load OpenAI API Key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing. Set it as an environment variable.")

# Web Scraping: Fetch Data from URL
url = "https://brainlox.com/courses/category/technical"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    full_text = soup.get_text()
else:
    raise RuntimeError(f"Failed to fetch webpage: {response.status_code}")

# Text Splitting for Better Embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.create_documents([full_text])

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Convert Documents into Vector Embeddings
vectorstore = FAISS.from_documents(documents, embeddings)

# Save FAISS index
vectorstore.save_local("faiss_index")

# Initialize Flask App
app = Flask(__name__)

# Load FAISS index
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Define Chatbot Function
def chatbot_response(query):
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), retriever=retriever)
    response = qa.run(query)
    return response

# Create an API Endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    response = chatbot_response(user_input)
    return jsonify({"response": response})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
