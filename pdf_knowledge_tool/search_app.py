
import streamlit as st
from tinydb import TinyDB, Query
import chromadb
from pathlib import Path
import json
import subprocess
import os
import sys

# Configuration
st.set_page_config(layout="wide", page_title="PDF Knowledge Search")

# Paths
DATA_DIR = Path("./data")
METADATA_DB_PATH = DATA_DIR / "metadata.json"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
COLLECTION_NAME = "pdf_knowledge"

# Initialize Connections (Cached)
@st.cache_resource
def get_db_connections():
    # Chroma
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    # TinyDB
    metadata_db = TinyDB(METADATA_DB_PATH)
    
    return collection, metadata_db

try:
    collection, metadata_db = get_db_connections()
    db_connected = True
except Exception as e:
    st.error(f"Database Connection Failed: {e}")
    db_connected = False

# UI Layout
st.title("🔎 Enterprise PDF Knowledge Search")

if not db_connected:
    st.warning("Please run `ingest.py` to initialize databases first.")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["🔎 Search", "📤 Upload & Index"])

# --- TAB 1: SEARCH ---
with tab1:
    st.subheader("Search Knowledge Base")
    query = st.text_input("Enter search query:", placeholder="e.g. 'quarterly revenue growth'")

    if query:
        with st.spinner("Searching..."):
            # 1. Search Vector DB
            results = collection.query(
                query_texts=[query],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'][0]:
                st.info("No results found.")
            else:
                # 2. Display Results
                for i, (doc_text, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
                    
                    pdf_id = metadata.get('pdf_id', 'Unknown')
                    source_type = metadata.get('source', 'text')
                    page_num = metadata.get('page', '?')
                    
                    with st.expander(f"Result {i+1} | Source: {pdf_id} (Page {page_num}) | Score: {distance:.2f}", expanded=True):
                        
                        # fetch full doc metadata for context if needed
                        full_doc = metadata_db.get(Query().pdf_id == pdf_id)
                        
                        if source_type == 'table':
                            st.markdown("**Found in Table:**")
                            # Try to find the specific table HTML
                            chunk_index = metadata.get('chunk_index')
                            if full_doc and 'tables' in full_doc:
                                 tables = full_doc['tables']
                                 if chunk_index is not None and chunk_index < len(tables):
                                     st.html(tables[chunk_index]['html'])
                                 else:
                                     st.text(doc_text)
                        else:
                            st.markdown(doc_text)
                        
                        if full_doc:
                            st.caption(f"Filename: {full_doc.get('filename')}")

# --- TAB 2: UPLOAD ---
with tab2:
    st.subheader("Upload a PDF to Index")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    doc_id = st.text_input("Document ID (Unique)", placeholder="e.g. report_Q1_2025")
    skip_vlm = st.checkbox("Skip Image Analysis (Faster)", value=False)
    
    if st.button("Index PDF", type="primary"):
        if uploaded_file and doc_id:
            # Save temp file
            temp_dir = Path("./temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"File saved to {temp_path}. Starting ingestion pipeline...")
            
            # Construct command
            cmd = [sys.executable, "ingest.py", "--pdf", str(temp_path), "--output-id", doc_id]
            if skip_vlm:
                cmd.append("--skip-vlm")
            
            # Run CLI ingestion
            # Using st.status for better feedback
            with st.status("Indexing...", expanded=True) as status:
                st.write("Initializing pipeline...")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    status.update(label="Ingestion Complete!", state="complete", expanded=False)
                    st.success(f"✅ Successfully indexed '{doc_id}'")
                    # Clear cache to see new results immediately if possible (though chroma client is persistent)
                    # st.cache_resource.clear() 
                else:
                    status.update(label="Ingestion Failed", state="error", expanded=True)
                    st.error(f"❌ Error during ingestion:")
                    st.code(result.stderr)
        else:
            st.warning("Please provide both a PDF file and a Document ID.")

# Sidebar Stats
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Knowledge Stats")
try:
    st.sidebar.metric("Total Vectors", collection.count())
    st.sidebar.metric("Indexed Documents", len(metadata_db.all()))
except:
    pass
