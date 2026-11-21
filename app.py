import os
import tempfile
import streamlit as st
from rag import index_pdf, search_documents, generate_answer

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")
st.write(
    "Upload a PDF, index it into ChromaDB, "
    "and chat with the document using semantic search and AI."
)

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )
    if uploaded_file:
        if st.button("Index PDF"):
            with st.spinner("Indexing document..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:
                    tmp.write(
                        uploaded_file.getbuffer()
                    )
                    pdf_path = tmp.name
                try:
                    number_of_chunks = index_pdf(
                        pdf_path
                    )
                    st.session_state[
                        "indexed"
                    ] = True
                    st.session_state[
                        "filename"
                    ] = uploaded_file.name
                    st.session_state[
                        "chunks"
                    ] = number_of_chunks
                    st.success(
                        f"PDF indexed successfully! "
                        f"{number_of_chunks} chunks created."
                    )
                except Exception as e:
                    st.error(
                        f"Error: {str(e)}"
                    )

if "indexed" not in st.session_state:
    st.info(
        "Upload a PDF from the sidebar and click "
        "'Index PDF' to begin."
    )
else:
    st.success(
        f"Indexed: {st.session_state['filename']} "
        f"({st.session_state['chunks']} chunks)"
    )
    st.divider()
    
    st.subheader("Ask a question")
    question = st.text_input(
        "Enter your question:"
    )
    
    if st.button("Search & Generate Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and generating answer..."):
                results = search_documents(
                    question,
                    n_results=3
                )
                documents = results["documents"][0]
                distances = results["distances"][0]
                
                answer = generate_answer(
                    question,
                    documents
                )
                
            st.subheader("Answer")
            st.write(answer)
            
            st.subheader("Retrieved Sources")
            for i, (document, distance) in enumerate(
                zip(documents, distances)
            ):
                with st.expander(f"Source {i + 1}"):
                    st.write(document)
                    st.caption(
                        f"Vector distance: {distance:.4f}"
                    )