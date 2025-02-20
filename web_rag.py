import gradio as gr
from mlc_llm import MLCEngine
from mlc_llm.serve.config import EngineConfig
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from bs4 import BeautifulSoup
import requests
import json
import os
import re

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
# Initialize the MLCEngine
model = "HF://mlc-ai/Qwen2.5-Coder-3B-Instruct-q4f16_1-MLC"
engine = MLCEngine(model, engine_config=EngineConfig(prefill_chunk_size=4000))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

url_pattern = re.compile(r"https?://\S+|www\.\S+")

def extract_html_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    data = [[(f"h{i}", tag.text.strip()) for tag in soup.find_all(f"h{i}")] for i in range(1, 7)]
    flattened_list = [item for sublist in data for item in sublist]

    return flattened_list, response.text


def prompt_response(query, history):
    retrieved_docs = retriever.get_relevant_documents(query)
    extracted_data = "\n\n".join([doc.page_content for doc in retrieved_docs])

    response = engine.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant built to answer questions using the provided context."},
            {"role": "user", "content": f"Context:\n{extracted_data}\n\nQuestion: {query}"}
        ],
        model=model,
        stream=False
    )
    
    output = response.choices[0].message.content

    if response:
        return output
    else:
        return "No data found!"

def process_url(url):
    global retriever
    url_found = url_pattern.findall(url)
    if not url_found:
        return "❌ URL not found!"
    
    headers_to_split_on, extracted_html = extract_html_data(url_found[0])
    
    #Document-structured based splitting
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        separators=["\n\n", "\n", ". ", "! ", "? "],
        max_chunk_size=50,
        preserve_images=True,
        preserve_videos=True,
        elements_to_preserve=["table", "ul", "ol", "code"],
        denylist_tags=["script", "style", "head"]
    )

    documents = splitter.split_text(extracted_html)
    # print(documents)

    index = FAISS.from_documents(documents, embeddings)
    retriever = index.as_retriever()
    return f"✅ Processing completed for: {url}"



with gr.Blocks() as demo:
    gr.Markdown("## Web Content Q&A tool")

    with gr.Row():
        url_input = gr.Textbox(label="Enter URL")
        submit_button = gr.Button("Submit")

    result_output = gr.Textbox(label="Result", interactive=False)

    submit_button.click(process_url, inputs=[url_input], outputs=[result_output])

    gr.ChatInterface(prompt_response, fill_height=True)


demo.launch(share=True)