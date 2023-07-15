import random
import torch
import streamlit as st
import openai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import time

# Load the environment variables from the .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")



def read_pdf(file):
    # Initialize a PDF file reader object
    pdf_reader = PdfReader(file)
    
    # Initialize an empty string to hold the extracted text
    text = ""
    
    # Loop through each page in the PDF file and extract the text
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

from transformers import AutoTokenizer, AutoModel

def embed_text(text):
    # Load the BERT model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # Tokenize the text and truncate it to the maximum length that the BERT model can handle
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Get the embeddings
    outputs = model(**inputs)

    # The embeddings are stored in the last hidden state
    embeddings = outputs.last_hidden_state

    # We take the mean of the embeddings from all tokens, resulting in one embedding vector
import random
import torch
import streamlit as st
import openai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import time

# Load the environment variables from the .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def send_request_to_gpt4(text):
    # Load the BERT model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize the text and split it into chunks of 4096 tokens
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + 4096] for i in range(0, len(tokens), 4096)]

    responses = []

    # Send a request to the GPT-4 model with each chunk
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        for i in range(5):  # Retry up to 5 times
            try:
                response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": "You are a AI assistant whose expertise is reading and summarizing scientific papers. You are given a query, a series of text embeddings and the title from a paper in order of their cosine similarity to the query. You must take the given embeddings and return a very detailed summary of the paper in the language of the query:"}, {"role": "user", "content": chunk_text}])
                responses.append(response['choices'][0]['message']['content'])
                time.sleep(0.1)  # Add a delay between each request to avoid hitting the rate limit
                break
            except openai.error.RateLimitError as e:  # Catch the RateLimitError
                if i < 4:  # If not the last retry attempt
                    time.sleep((2 ** i) + (random.randint(0, 1000) / 1000))  # Exponential backoff with jitter
                else:
                    raise e  # If the last retry attempt, re-raise the exception

    return " ".join(responses)
    files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)
    
    if files:
        for file in files:
            # Read the PDF file and extract the text
            text = read_pdf(file)
            
            # Embed the extracted text
            embedded_text = embed_text(text)
            
            # Send a request to the GPT-4 model and get the response
            response = send_request_to_gpt4(text)
            
            # Display the response
            st.write(response)
def main():
    files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)
    
    if files:
        # Initialize an empty list to hold the text from all uploaded files
        all_text = []
        
        for file in files:
            # Read the PDF file and extract the text
            text = read_pdf(file)
            
            # Add the extracted text to the list
            all_text.append(text)
        
        # Join all the text into one string
        all_text = " ".join(all_text)
        
        # Initialize an empty list to hold the chat history
        chat_history = []
        
        # Create a text input field for the user to enter their messages
        user_message = st.text_input("Enter your message:")
        
        # Create a button to send the message
        if st.button("Send"):
            # Add the user's message to the chat history
            chat_history.append({"role": "user", "content": user_message})
            
            # Send a request to the GPT-4 model with the user's message and the content of the uploaded PDF files
            response = send_request_to_gpt4(all_text + " " + user_message)
            
            # Add the model's response to the chat history
            chat_history.append({"role": "gpt-4", "content": response})
            
            # Display the chat history
            for message in chat_history:
                if message["role"] == "user":
                    st.write("User: " + message["content"])
                else:
                    st.write("GPT-4: " + message["content"])

if __name__ == "__main__":
    main()
