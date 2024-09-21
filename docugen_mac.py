import subprocess
import gradio as gr
from PyPDF2 import PdfReader
import docx2txt
import re

# Function to load and split PDF efficiently
def load_and_split_pdf(pdf_path, chunk_size=2000):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text
    
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Function to load and split DOCX efficiently
def load_and_split_docx(docx_path, chunk_size=2000):
    text = docx2txt.process(docx_path)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Detect the document type (text, numeric, or mixed)
def detect_document_type(text_chunks):
    numeric_count = 0
    text_count = 0

    for chunk in text_chunks:
        if re.search(r'\d', chunk):
            numeric_count += 1
        else:
            text_count += 1

    if numeric_count > text_count:
        return 'numeric'
    elif text_count > numeric_count:
        return 'text'
    else:
        return 'mixed'

# Automatically provide the best answer based on the user's prompt and document content
def get_best_answer(user_query, text_chunks):
    context = "\n\n".join(text_chunks)
    prompt = f"Answer the following query concisely based on the provided context:\n\n{user_query}\n\nContext: {context[:2000]}"  # Limit context to 2000 characters
    
    # Use subprocess to run Ollama locally
    result = subprocess.run(
        ["ollama", "run", "llama3.1:8b-instruct-fp16", "--prompt", prompt],
        capture_output=True,
        text=True
    )
    
    # Return the generated answer
    return result.stdout.strip()

# Get a summary of the document
def get_summary(text_chunks):
    prompt = "Summarize the following document:\n\n" + "\n\n".join(text_chunks[:2000])
    
    # Use subprocess to run Ollama locally
    result = subprocess.run(
        ["ollama", "run", "llama3.1:8b-instruct-fp16", "--prompt", prompt],
        capture_output=True,
        text=True
    )
    
    return result.stdout.strip()

# Refine document content
def refine_document_content(text_chunks):
    prompt = "Refine the following content for clarity and coherence:\n\n" + "\n\n".join(text_chunks[:2000])
    
    # Use subprocess to run Ollama locally
    result = subprocess.run(
        ["ollama", "run", "llama3.1:8b-instruct-fp16", "--prompt", prompt],
        capture_output=True,
        text=True
    )
    
    return result.stdout.strip()

# Check compliance of the document
def check_compliance(text_chunks):
    prompt = "Check the following document for compliance issues and highlight potential risks:\n\n" + "\n\n".join(text_chunks[:2000])
    
    # Use subprocess to run Ollama locally
    result = subprocess.run(
        ["ollama", "run", "llama3.1:8b-instruct-fp16", "--prompt", prompt],
        capture_output=True,
        text=True
    )
    
    return result.stdout.strip()

# Gradio interface function
def chatbot_interface(document, user_query, action):
    try:
        if document.name.endswith('.pdf'):
            text_chunks = load_and_split_pdf(document.name)
        elif document.name.endswith('.docx'):
            text_chunks = load_and_split_docx(document.name)
        else:
            return "Unsupported file type."

        document_type = detect_document_type(text_chunks)

        if action == "Get Summary":
            return get_summary(text_chunks)
        elif action == "Refine Content":
            return refine_document_content(text_chunks)
        elif action == "Check Compliance":
            return check_compliance(text_chunks)
        else:
            # Automatically provide best answer based on user's query in the textbox
            return get_best_answer(user_query, text_chunks)

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error: Something went wrong. Please try again later."

# Gradio Interface
interface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.File(label="Upload Document"), gr.Textbox(lines=2, placeholder="Enter your question here..."), gr.Radio(choices=["Get Summary", "Refine Content", "Check Compliance"], label="Choose Action")],
    outputs="text",
    title="Document Analysis Assistant",
    description="Upload a PDF or DOCX document and ask a question, get a summary, refine content, or check compliance.",
    live=True  # Enable live interaction for real-time updates
)

# Launch the interface
interface.launch()

