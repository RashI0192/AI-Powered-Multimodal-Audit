graph TD
    classDef query fill:#f9f,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef process fill:#bbf,stroke:#333,stroke-width:1px,color:#000;
    classDef decision fill:#fea,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef output fill:#bfb,stroke:#333,stroke-width:1px,color:#000;

    A[User asks a question]:::query --> B{Is query about image/chart?}:::decision

    B -- No --> C[Similarity search on text & tables (FAISS)]:::process
    C --> D[Retrieve top text/table documents]:::process
    D --> E[Generate answer using ChatGroq LLM (llama3-70b-8192)]:::process

    B -- Yes --> F[Similarity search on image captions (FAISS)]:::process
    F --> G{Is relevant image caption found?}:::decision
    
    G -- No --> C
    G -- Yes --> H[Verify image is a chart with chart-recognizer]:::process

    H --> I{Is image confirmed as chart?}:::decision
    I -- No --> C
    I -- Yes --> J[Run BLIP-VQA on original image + user query]:::process
    J --> K[Extract numbers from image using pytesseract OCR]:::process
    K --> L[Detect visual trend via OpenCV (HSV, Canny, HoughLines, slope)]:::process
    L --> M[Return combined VQA answer + OCR data + trend analysis]:::output
