import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Function to load and cache the summarization model
@st.cache_resource
def load_summarization_pipeline():
    return pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Function to preprocess and chunk text
def preprocess_text(text, max_chunk_size=512):
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
    return chunks

# Streamlit UI
st.title("MHU PDF Summarization App")
st.write("Upload a PDF file to generate a concise summary (200–300 words)!")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    text = " ".join(page.extract_text() for page in pdf_reader.pages)

    if text.strip():
        st.success("PDF text successfully extracted!")

        # Show extracted text in a collapsible section
        with st.expander("Extracted PDF Text"):
            st.write(text)

        # Load summarization pipeline
        summarization_pipeline = load_summarization_pipeline()

        # Preprocess and chunk text
        chunks = preprocess_text(text)
        st.write(f"PDF content split into {len(chunks)} chunks for summarization.")

        # Generate summary for each chunk
        with st.spinner("Generating summary..."):
            summaries = summarization_pipeline(" ".join(chunks), max_length=300, min_length=200, truncation=True)

        # Display the summary
        st.subheader("Summary:")
        st.write(summaries[0]["summary_text"])
    else:
        st.error("No text could be extracted from the PDF. Please upload a valid PDF.")
else:
    st.info("Please upload a PDF to start.")
