# layout.py
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

def render_sidebar(image_docs, table_docs):
    st.sidebar.header("Images Detected")
    for i, doc in enumerate(image_docs):
        img = Image.open(BytesIO(base64.b64decode(doc.metadata["image_base64"])))
        st.sidebar.image(img, caption=doc.metadata.get("caption", f"Image {i+1}"), use_column_width=True)

    st.sidebar.header("Tables Detected")
    for table in table_docs:
        st.sidebar.text(table.page_content[:300])

def render_pdf_uploader():
    st.title("PDF Q&A Chatbot (Text, Tables, Charts)")
    uploaded = st.file_uploader("Upload your PDF report", type=["pdf"])
    return uploaded

def render_chat_ui(answer_func):
    st.markdown("---")
    st.subheader("Ask a question about the PDF")
    user_q = st.text_input("Enter your question")
    if user_q:
        with st.spinner("Thinking....."):
            st.markdown(answer_func(user_q))
