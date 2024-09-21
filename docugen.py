import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from PyPDF2 import PdfReader
import docx2txt
import numpy as np
import re

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Set pad_token to eos_token if it's missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model with FP16
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,  # Use FP16
    device_map="auto"  # Automatically move model to GPU
)

# Function to load and split PDF
def load_and_split_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_chunks.extend(text.split("\n\n"))  # Split into chunks by paragraphs
    return text_chunks

# Function to load and split DOCX
def load_and_split_docx(docx_path):
    text = docx2txt.process(docx_path)
    return text.split("\n\n")

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

# Get the best answer using the local LLaMA model
def get_best_answer(user_query, text_chunks):
    context = "\n\n".join(text_chunks)
    prompt = f"Answer the following query concisely based on the provided context:\n\n{user_query}\n\nContext: {context[:2000]}"  # Limit context to 2000 characters
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")

    with torch.no_grad():
        response = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=500,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
    return generated_text.strip()

# Refine document content
def refine_document_content(text_chunks):
    prompt = "Refine the following content for clarity and coherence:\n\n" + "\n\n".join(text_chunks[:2000])
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")

    with torch.no_grad():
        response = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=500,
            pad_token_id=tokenizer.eos_token_id
        )

    refined_text = tokenizer.decode(response[0], skip_special_tokens=True)
    return refined_text.strip()

# Check compliance of the document
def check_compliance(text_chunks):
    prompt = "Check the following document for compliance issues and highlight potential risks:\n\n" + "\n\n".join(text_chunks[:2000])
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")

    with torch.no_grad():
        response = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=500,
            pad_token_id=tokenizer.eos_token_id
        )

    compliance_text = tokenizer.decode(response[0], skip_special_tokens=True)
    return compliance_text.strip()

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

        if action == "Get Answer":
            return get_best_answer(user_query, text_chunks)
        elif action == "Refine Content":
            return refine_document_content(text_chunks)
        elif action == "Check Compliance":
            return check_compliance(text_chunks)

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error: Something went wrong. Please try again later."

# Gradio Interface
interface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.File(label="Upload Document"), gr.Textbox(lines=2, placeholder="Enter your question here..."), gr.Radio(choices=["Get Answer", "Refine Content", "Check Compliance"], label="Choose Action")],
    outputs="text",
    title="Document Analysis Assistant",
    description="Upload a PDF or DOCX document and ask a question, refine content, or check compliance.",
    live=False
)

# Launch the interface
interface.launch()
