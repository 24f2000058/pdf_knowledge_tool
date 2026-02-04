
import os
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Third-party imports
from tinydb import TinyDB, Query
import chromadb
import config
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from schemas import DocumentMetadata, ChunkMetadata
import hashlib  # Ensure hashlib is imported

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ingest.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("./data")
IMAGES_DIR = DATA_DIR / "images"
METADATA_DB_PATH = DATA_DIR / "metadata.json"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
COLLECTION_NAME = "pdf_knowledge"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

class IngestionPipeline:
    def __init__(self, skip_vlm=False):
        # Configure Pipeline Options for Previews
        pipeline_options = PdfPipelineOptions()
        if config.ENABLE_PAGE_PREVIEWS:
            pipeline_options.generate_page_images = True
            pipeline_options.images_scale = 1.0 # Standard resolution
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Initialize Databases
        logger.info("Initializing Databases...")
        self.metadata_db = TinyDB(METADATA_DB_PATH)
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of the file for delta updates."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def check_exists(self, pdf_hash: str) -> bool:
        """Check if a file with this hash has already been indexed."""
        doc = self.metadata_db.get(Query().file_hash == pdf_hash)
        return doc is not None

    def process_pdf(self, pdf_path: str, output_id: str):
        # 0. Delta Check (Hash)
        file_hash = self.calculate_file_hash(pdf_path)
        if self.check_exists(file_hash):
             logger.warning(f"Duplicate file detected (Hash: {file_hash}). Skipping ingestion.")
             return

        logger.info(f"Processing PDF: {pdf_path} (ID: {output_id}) | Hash: {file_hash}")
        
        # 1. Parse PDF
        try:
            conv_res = self.doc_converter.convert(pdf_path)
            doc = conv_res.document
        except Exception as e:
            logger.error(f"Docling conversion failed: {e}")
            raise e

        # Save Page Previews (if enabled and present)
        if config.ENABLE_PAGE_PREVIEWS:
            saved_images_count = 0
            for i, page in enumerate(doc.pages.values()):
                if saved_images_count >= config.PREVIEW_MAX_PAGES:
                    break
                    
                if hasattr(page, 'image') and page.image and page.image.pil_image:
                    image_path = IMAGES_DIR / f"{output_id}_page_{page.page_no}.jpg"
                    try:
                        page.image.pil_image.save(image_path, "JPEG")
                        saved_images_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to save preview for page {page.page_no}: {e}")
            
            if saved_images_count > 0:
                logger.info(f"Saved {saved_images_count} page previews.")
        
        markdown_text = doc.export_to_markdown()
        logger.debug(f"Markdown extracted. Length: {len(markdown_text)}")
        
        tables_data = []

        for i, table in enumerate(doc.tables):
            try:
                html_content = table.export_to_html()
                table_text = table.export_to_dataframe().to_string() 
                
                if html_content and len(html_content.strip()) > 0:
                    tables_data.append({
                        "index": i,
                        "html": html_content,
                        "summary": table_text,
                        "page": table.prov[0].page_no if table.prov else 1
                    })
                else:
                    logger.warning(f"Table {i} export resulted in empty HTML. Skipping.")
            except Exception as e:
                logger.warning(f"Failed to export table {i}: {e}")

        # 3. Chunking (Native Docling with Provenance)
        # Using default HybridChunker configuration which is generally robust
        chunker = HybridChunker()
        
        chunks = []
        doc_chunks = list(chunker.chunk(doc))
        logger.info(f"DEBUG: Docling generated {len(doc_chunks)} chunks")
        
        for i, chunk in enumerate(doc_chunks):
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue
            
            # Extract Page Number from Provenance
            # Docling chunks can span pages, usually take the first item's page
            page = 1
            if chunk.meta.doc_items:
                # prov is a list of ProvenanceItem
                first_item = chunk.meta.doc_items[0]
                if hasattr(first_item, 'prov') and first_item.prov:
                     # prov is list of Prov items, take first
                     page = first_item.prov[0].page_no
            elif hasattr(chunk.meta, 'provenance') and chunk.meta.provenance: 
                # Fallback for different docling constraints
                page = chunk.meta.provenance[0].page_no

            logger.info(f"DEBUG: Chunk {i} | Page: {page} | Len: {len(chunk_text)}")

            # Use Pydantic Schema for Metadata
            meta = ChunkMetadata(
                pdf_id=output_id,
                chunk_index=i,
                page_number=page,
                data_source="docling_text"
            )
            
            chunks.append({
                "id": f"{output_id}_text_{i}",
                "text": chunk_text,
                "metadata": meta.model_dump(exclude_none=True)
            })
        
        for i, tbl in enumerate(tables_data):
            # Add tables as chunks
            chunks.append({
                "id": f"{output_id}_table_{i}",
                "text": f"Table Content:\n{tbl['summary']}",
                 "metadata": ChunkMetadata(
                    pdf_id=output_id,
                    chunk_index=i,
                    data_source="html_table",
                    page_number=tbl['page']
                ).model_dump(exclude_none=True)
            })

        # 4. Store in Chroma
        if chunks:
            ids = [c["id"] for c in chunks]
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Stored {len(chunks)} chunks in ChromaDB.")

        # 5. Store Metadata in TinyDB
        # 5. Store Metadata in TinyDB (Schema Validated)
        doc_meta = DocumentMetadata(
            id=output_id,
            filename=os.path.basename(pdf_path),
            file_hash=file_hash,
            total_chunks=len(chunks),
            tables=[t['html'] for t in tables_data],
            images=[]
        )
        self.metadata_db.insert(doc_meta.model_dump(mode='json'))
        logger.info(f"Stored valid metadata for {output_id} in TinyDB.")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF into Knowledge Base")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output-id", required=True, help="Unique ID for this document")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM loading (images won't be captioned)")
    
    args = parser.parse_args()
    
    pipeline = IngestionPipeline(skip_vlm=args.skip_vlm)
    
    # Remove legacy ID duplicate check here, as we do hash check inside process_pdf now
    # But we can keep an optional ID check if desired, but hash is safer.
    
    start_time = time.time()

    try:
        pipeline.process_pdf(args.pdf, args.output_id)
        logger.info("Ingestion Complete.")
    except Exception as e:
        logger.error(f"Ingestion Failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
