# pdf_utils.py
import base64
from io import BytesIO
from PIL import Image
import pytesseract
from transformers import pipeline
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf

def parse_pdf(file_path):
    return partition_pdf(filename=file_path, strategy="hi_res", infer_table_structure=True,
                         extract_image_block_types=["Image"], extract_image_block_to_payload=True)

def process_chunks(chunks):
    text_chunks, table_chunks, image_chunks = [], [], []
    for chunk in chunks:
        if chunk.category == "Table":
            table_chunks.append(chunk)
        elif chunk.category == "Image":
            image_chunks.append(chunk)
        elif hasattr(chunk, "text") and chunk.text.strip():
            text_chunks.append(chunk)
    return text_chunks, table_chunks, image_chunks

def process_images(image_chunks):
    blip = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    image_docs = []
    for i, chunk in enumerate(image_chunks):
        b64 = chunk.metadata["image_base64"]
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        caption = blip(img)[0]["generated_text"]
        ocr_text = pytesseract.image_to_string(img)
        numbers = [s for s in ocr_text.split() if s.replace('.', '', 1).isdigit()]
        image_docs.append(Document(
            page_content=caption + "\nOCR Numbers: " + ", ".join(numbers),
            metadata={"type": "image", "image_base64": b64, "caption": caption, "source": f"image_chunk_{i}"}
        ))
    return image_docs
