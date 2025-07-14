This folder contains 2 notebooks: 
1. approach_testing_and_debugging
- This folder contains experimentation notebooks used to evaluate multiple strategies for building a robust PDF-based question-answering system. The goal was to identify performance bottlenecks, test solutions, and refine the pipeline before final deployment.

### Contents
- Iterative testing of PDF parsing techniques (unstructured, OCR with Tesseract)
- Visual data extraction experiments (image classification, chart trend detection, BLIP for VQA)
- Rate limit handling logic for reliable API-based summarization
- Notes and logs on edge cases, hallucinations, and fallback mechanisms
### Key Challenges Explored 
1. Scanned vs. Digital PDFs
- Different extraction logic was required for scanned documents (using OCR) versus digitally generated PDFs (with structured tags). This was handled using conditional pipelines and preprocessing.
2. Image and Chart Understanding
- Interpreting visual elements such as charts required:
1. Classifying image types using a chart-recognizer model
2. Extracting numerical and textual data using OCR
3. Analyzing trend lines and answering image-based questions via BLIP
4. Rate Limit Resilience - a retry-and-wait mechanism was implemented to gracefully handle rate limits during summarization with LangChain and external APIs.

2. multimodal_agent: The final version of the agent logic, used to build the Streamlit-based demo.
  
3. sample_pdfs/: A collection of test PDFs focused on research text, tables, and chart-heavy documents to evaluate the generalization capability of the pipeline.

