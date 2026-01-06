
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
from docling.document_converter import DocumentConverter
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import hashlib
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from schemas import DocumentMetadata, ChunkMetadata

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
        self.doc_converter = DocumentConverter()
        
        # Initialize Databases
        logger.info("Initializing Databases...")
        self.metadata_db = TinyDB(METADATA_DB_PATH)
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        
        # Initialize VLM (Florence-2)
        self.vlm_ready = False
        if not skip_vlm:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            logger.info(f"Loading Florence-2 on {self.device}...")
            try:
                # Use a smaller model for speed locally
                model_id = "microsoft/Florence-2-base"
                self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                # "eager" attention implementation often bypasses the SDPA check causing the error
                # Use standard 'dtype' argument
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    dtype=self.dtype, 
                    trust_remote_code=True,
                    attn_implementation="eager" 
                ).to(self.device)
                self.vlm_ready = True
            except Exception as e:
                logger.error(f"Failed to load VLM: {e}. Image captioning will be skipped.", exc_info=True)
        else:
            logger.info("Skipping VLM loading as requested.")

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

    def run_inference(self, image_path: Path) -> str:
        if not self.vlm_ready:
            return "Image captioning unavailable."
        
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.dtype)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
            return parsed_answer.get(prompt, "No caption generated.")
            
        except Exception as e:
            logger.error(f"Error captioning image {image_path}: {e}")
            return "Error generating caption."

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

        # 3. Chunking
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = []
        raw_chunks = splitter.split_text(markdown_text)
        logger.info(f"DEBUG: Raw chunks count: {len(raw_chunks)}")
        
        for i, content in enumerate(raw_chunks):
            logger.info(f"DEBUG: Chunk {i} len: {len(content.strip())} | Content: {content[:20]}...")
            chunk_text = content.strip()
            if len(chunk_text) > 5:
                # Use Pydantic Schema for Metadata
                meta = ChunkMetadata(
                    pdf_id=output_id,
                    chunk_index=i,
                    page_number=None, # Docling text splitter doesn't preserve exact page yet easily
                    data_source="docling_text"
                )
                
                chunks.append({
                    "id": f"{output_id}_text_{i}",
                    "text": content,
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
