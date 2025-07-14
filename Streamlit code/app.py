# app.py
import streamlit as st
from layout import render_pdf_uploader, render_sidebar, render_chat_ui
from pdf_utils import parse_pdf, process_chunks, process_images
from rag_engine import build_vectorstore, setup_rag_chain, build_answer_function
from langchain.schema import Document

st.set_page_config(page_title="PDF Visual Q&A Bot", layout="wide")
uploaded_file = render_pdf_uploader()

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF uploaded and processed.")

    chunks = parse_pdf("temp.pdf")
    text_chunks, table_chunks, image_chunks = process_chunks(chunks)

    text_docs = [Document(page_content=c.text, metadata={"source": f"text_{i}"}) for i, c in enumerate(text_chunks)]
    table_docs = [Document(page_content=c.text, metadata={"source": f"table_{i}"}) for i, c in enumerate(table_chunks)]
    image_docs = process_images(image_chunks)

    all_docs = text_docs + table_docs + image_docs
    db = build_vectorstore(all_docs)
    rag_chain = setup_rag_chain()
    answer_func = build_answer_function(db, rag_chain)

    render_sidebar(image_docs, table_docs)
    render_chat_ui(answer_func)
