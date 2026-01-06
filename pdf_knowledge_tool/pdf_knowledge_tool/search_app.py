

import streamlit as st
from tinydb import TinyDB, Query
import chromadb
from pathlib import Path
import json
import subprocess
import os
import sys
import logging
import config
import models
import retrieval
from rank_bm25 import BM25Okapi

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

# Navigation (Persistent State)
mode = st.radio("Mode", ["🔎 Search", "📤 Upload & Index"], horizontal=True, label_visibility="collapsed")

# --- SIDEBAR: SEARCH HISTORY & FILTERS ---
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("🔍 Filters")
    # Filters
    all_docs = metadata_db.all()
    all_ids = [d.get('pdf_id', 'unknown') for d in all_docs]
    selected_ids = st.multiselect("Filter by Document:", all_ids)
    
    st.divider()
    st.subheader("📜 Search History")
    for h in st.session_state.history[-5:]: # Show last 5
        if st.button(h, key=f"hist_{h}"):
            st.session_state.query_input = h
            st.rerun()

# --- VIEW 1: SEARCH ---
if mode == "🔎 Search":
    st.subheader("Search Knowledge Base")
    
    # Use session state for query to allow history clicks
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
        
    query = st.text_input("Enter search query:", value=st.session_state.query_input, placeholder="e.g. 'quarterly revenue growth'")

    # Initialize LLM Service (Singleton/Cached)
    @st.cache_resource
    def get_llm_service():
        return models.LLMService()

    if query:
        # Save to history if new
        if query not in st.session_state.history:
            st.session_state.history.append(query)
            
        if collection.count() == 0:
            st.warning("The knowledge base is currently empty. Please upload and index a PDF first!")
        else:
            llm_service = get_llm_service()
            
            # 1. Query Analysis & Expansion
            with st.status("Thinking...", expanded=False) as status:
                k_retrieval = retrieval.classify_query(query)
                status.write(f"Classified query. Target chunks: {k_retrieval}")
                
                expanded_query = retrieval.expand_query(query, llm_service)
                if expanded_query != query:
                    status.write(f"Expanded query terms: '{expanded_query}'")
                
                # 2. Retrieval Pipeline
                status.write("Searching database...")
                candidates = retrieval.full_hybrid_search(expanded_query, collection, k=config.TOP_N_RETRIEVAL)
                
                # Filter by Document ID
                if selected_ids:
                    candidates = [c for c in candidates if c['metadata'].get('pdf_id') in selected_ids]
                
                # 3. Neighbor Expansion
                top_k_candidates = candidates[:k_retrieval]
                context_chunks_raw = retrieval.expand_neighbors(collection, top_k_candidates)
                
                # 4. Post-Retrieval Filtering
                context_chunks = retrieval.filter_chunks(context_chunks_raw, query)
                status.update(label="Retrieval Complete", state="complete")
            
            if not context_chunks:
                st.info("No relevant information found.")
            else:
                # 5. Context Construction
                context_text_list = []
                for chunk in context_chunks:
                    meta = chunk['metadata']
                    pid = meta.get('pdf_id', 'unknown')
                    pg = meta.get('page', '?')
                    text = chunk['document']
                    context_text_list.append(f"[Source: {pid}, Page: {pg}]\n{text}")
                
                full_context = "\n\n".join(context_text_list)
                max_chars = config.MAX_CONTEXT_TOKENS * 4
                if len(full_context) > max_chars:
                    full_context = full_context[:max_chars] + "...(truncated)"
                
                # 6. Answer Generation
                st.markdown("### 🤖 AI Answer")
                with st.spinner("Generating answer..."):
                    answer = llm_service.generate_answer(full_context, query)
                    st.write(answer)
                
                # Export Button
                st.download_button(
                    label="💾 Export Answer (JSON)",
                    data=json.dumps({"query": query, "answer": answer, "context": context_text_list}, indent=2),
                    file_name="answer_export.json",
                    mime="application/json"
                )
                
                st.divider()
                
                # 7. Source Attribution
                with st.expander("📚 View Supporting Sources", expanded=False):
                    for i, res in enumerate(context_chunks):
                            meta = res['metadata']
                            doc_text = res['document']
                            score = res['score']
                            
                            pdf_id = meta.get('pdf_id', 'Unknown')
                            source_type = meta.get('source', 'text')
                            page_num = meta.get('page', '?')
                            
                            st.markdown(f"**Source {i+1}** | {pdf_id} (Page {page_num}) | Score: {score:.2f}")
                            
                            # Full Doc Context (for Images/Tables)
                            full_doc = metadata_db.get(Query().pdf_id == pdf_id)
                            
                            if source_type == 'table':
                                st.caption("Found in Table:")
                                chunk_index = meta.get('chunk_index')
                                if full_doc and 'tables' in full_doc:
                                    tables = full_doc['tables']
                                    if chunk_index is not None and chunk_index < len(tables):
                                        table_html = tables[chunk_index]['html']
                                        if table_html:
                                            st.html(table_html)
                                        else:
                                            st.text(doc_text)
                                    else:
                                        st.text(doc_text)
                            else:
                                st.text(doc_text)
                                # Image Preview (Fallback logic)
                                # In future: Check if page has image in images/ folder
                            
                            st.divider()

# --- VIEW 2: UPLOAD ---
elif mode == "📤 Upload & Index":
    st.subheader("Upload a PDF to Index")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    doc_id = st.text_input("Document ID", placeholder="e.g. Q1_Report")
    skip_vlm = st.checkbox("Skip Image Analysis", value=False)
    
    if st.button("Index PDF", type="primary"):
        if uploaded_file and doc_id:
            temp_dir = Path("./temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"Starting ingestion for {doc_id}...")
            cmd = [sys.executable, "ingest.py", "--pdf", str(temp_path), "--output-id", doc_id]
            if skip_vlm: cmd.append("--skip-vlm")
            
            with st.status("Indexing...", expanded=True) as status:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    status.update(label="Complete!", state="complete")
                    st.success("✅ Document Indexed.")
                else:
                    status.update(label="Failed", state="error")
                    st.error("❌ Ingestion Failed")
                    st.code(result.stderr)
        else:
            st.warning("Please provide file and ID.")

# Sidebar Stats
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Knowledge Stats")
try:
    st.sidebar.metric("Total Vectors", collection.count())
    st.sidebar.metric("Indexed Documents", len(metadata_db.all()))
except:
    pass
