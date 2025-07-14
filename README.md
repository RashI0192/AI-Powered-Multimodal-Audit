# AI-Powered-Multimodal-Audit

graph TD
    A[User asks a question] --> B{Is query about image/chart?}

    B -- No --> C[Similarity search on text & tables (FAISS)]
    C --> D[Retrieve top text/table documents]
    D --> E[Generate answer using ChatGroq LLM (llama3-70b-8192)]

    B -- Yes --> F[Similarity search on image captions (FAISS)]
    F --> G{Is relevant image caption found?}
    
    G -- No --> C
    G -- Yes --> H[Verify image is a chart with chart-recognizer]

    H --> I{Is image confirmed as chart?}
    I -- No --> C
    I -- Yes --> J[Run BLIP-VQA on original image + user query]
    J --> K[Extract numbers from image using pytesseract OCR]
    K --> L[Detect visual trend via OpenCV (HSV, Canny, HoughLines, slope)]
    L --> M[Return combined VQA answer + OCR data + trend analysis]
