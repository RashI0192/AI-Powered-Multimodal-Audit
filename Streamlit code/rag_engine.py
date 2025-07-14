# rag_engine.py
import base64
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import cv2
from env import GROQ_API_KEY

def build_vectorstore(all_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(all_docs, embeddings)

def setup_rag_chain():
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.3, api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant. Use only the provided context to answer the user's question.
The context may include articles, notes, tables, or reports.
Do not make up information. Be clear, accurate, and concise.

Context:
{context}

Question:
{question}

Answer:
""")
    return prompt | llm | StrOutputParser()

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
chart_classifier = pipeline("image-classification", model="StephanAkkerman/chart-recognizer")

def ask_blip_vqa_from_b64(b64, question):
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    inputs = processor(img, question, return_tensors="pt").to(vqa_model.device)
    output = vqa_model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def detect_colored_trend_line(b64):
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    open_cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(open_cv_img, cv2.COLOR_BGR2HSV)
    low_gray, high_gray = np.array([0, 0, 0]), np.array([180, 70, 255])
    mask = cv2.bitwise_not(cv2.inRange(hsv, low_gray, high_gray))
    colored = cv2.bitwise_and(open_cv_img, open_cv_img, mask=mask)
    gray = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines is None:
        return " No lines detected"
    slopes = [(y2 - y1) / (x2 - x1 + 1e-6) for [[x1, y1, x2, y2]] in lines if abs(x2 - x1) > 1]
    avg = np.mean(slopes)
    return " Upward" if avg < -0.2 else "Downward" if avg > 0.2 else "Flat"

query_model = SentenceTransformer("all-MiniLM-L6-v2")
example_embeds = query_model.encode(["What does the chart show?", "Describe the image", "Analyze the figure"], convert_to_tensor=True)

def is_query_about_image(q):
    q_embed = query_model.encode(q, convert_to_tensor=True)
    return torch.max(util.cos_sim(q_embed, example_embeds)).item() > 0.65

def build_answer_function(db, rag_chain):
    def answer(query):
        results = db.similarity_search(query, k=3)
        if is_query_about_image(query):
            for r in results:
                if r.metadata.get("type") == "image":
                    b64 = r.metadata["image_base64"]
                    caption = r.metadata.get("caption", "")
                    vqa = ask_blip_vqa_from_b64(b64, query)
                    trend = detect_colored_trend_line(b64)
                    ocr = pytesseract.image_to_string(Image.open(BytesIO(base64.b64decode(b64))))
                    nums = ", ".join([s for s in ocr.split() if s.replace('.', '', 1).isdigit()])
                    return f"""Image Match Found!
**Caption**: {caption}
**Answer**: {vqa}
**Trend**: {trend}
**Numbers**: {nums or 'None'}"""
        context = "\n\n".join([r.page_content for r in results])
        return rag_chain.invoke({"context": context, "question": query})
    return answer
