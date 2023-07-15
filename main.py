import random
import streamlit as st
import openai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import time
from transformers import AutoTokenizer
from roles import scientific_writing_specialist

# Load the environment variables from the .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_pdf(pdf_file):
    """
    Reads the text from a PDF file and generates a citation.

    Args:
        pdf_file: The PDF file to read.

    Returns:
        A tuple containing the extracted text and the generated citation.
    """
    # Initialize a PDF file reader object
    pdf_reader = PdfReader(pdf_file)
    
    # Initialize an empty string to hold the extracted text
    extracted_text = ""
    
    # Loop through each page in the PDF file and extract the text
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()
    
    # Generate a citation for the PDF file
    citation = generate_citation(extracted_text)
    
    return extracted_text, citation

def generate_citation(text):
    # For now, return a placeholder citation
    citation = "Placeholder citation for: " + text[:50] + "..."
    return citation

def send_request_to_gpt4(input_text, citation):
    """
    Sends a request to the GPT-4 model with the input text and citation.

    Args:
        input_text: The text to send to the GPT-4 model.
        citation: The citation to include in the response.

    Returns:
        The response from the GPT-4 model.
    """
    # Load the BERT model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize the input text and split it into chunks of 4096 tokens
    tokens = tokenizer.tokenize(input_text)
    token_chunks = [tokens[i:i + 4096] for i in range(0, len(tokens), 4096)]

    gpt_responses = []

    # Send a request to the GPT-4 model with each chunk
    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        for i in range(5):  # Retry up to 5 times
            try:
                response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": scientific_writing_specialist}, {"role": "user", "content": chunk_text}])
                gpt_responses.append(response['choices'][0]['message']['content'])  # type: ignore
                gpt_responses.append(citation)  # Add the citation to the response
                time.sleep(0.1)  # Add a delay between each request to avoid hitting the rate limit
                break
            except openai.error.RateLimitError as e:  # Catch the RateLimitError # type: ignore
                if i < 4:  # If not the last retry attempt
                    time.sleep((2 ** i) + (random.randint(0, 1000) / 1000))  # Exponential backoff with jitter
                else:
                    raise e  # If the last retry attempt, re-raise the exception

    return " ".join(gpt_responses)
def main():
    files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)
    
    if files:
        # Initialize an empty list to hold the text from all uploaded files
        all_text = []
        
        # Initialize an empty list to hold the citations
        all_citations = []
        
        for file in files:
            # Read the PDF file and extract the text and citation
            text, citation = read_pdf(file)
            
            # Add the extracted text to the list
            all_text.append(text)
            
            # Add the citation to the list
            all_citations.append(citation)
        
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
            
            # Display a loading message while the request is being processed
            with st.spinner('Processing...'):
                # Send a request to the GPT-4 model with the user's message, the content of the uploaded PDF files, and the citation
                response = send_request_to_gpt4(all_text + " " + user_message, citation) # type: ignore
            
            # Add the model's response to the chat history
            chat_history.append({"role": "gpt-4", "content": response})
            
            # Display a success message once the response is received
            if response:
                st.success('Done!')
            else:
                st.error('Something went wrong.')
            
            # Display the chat history
            for message in chat_history:
                if message["role"] == "user":
                    st.write("User: " + message["content"])
                else:
                    st.write("GPT-4: " + message["content"])
            
            # Display the citations as Streamlit captions
            for citation in all_citations:
                st.caption(citation)

if __name__ == "__main__":
    main()
