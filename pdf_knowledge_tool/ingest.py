
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
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            logger.info(f"Loading Florence-2 on {self.device}...")
            try:
                # Use a smaller model for speed locally
                model_id = "microsoft/Florence-2-base"
                self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
                self.vlm_ready = True
            except Exception as e:
                logger.error(f"Failed to load VLM: {e}. Image captioning will be skipped.")
        else:
            logger.info("Skipping VLM loading as requested.")

    def check_exists(self, output_id: str) -> bool:
        doc = self.metadata_db.get(Query().pdf_id == output_id)
        return doc is not None

    def run_inference(self, image_path: Path) -> str:
        if not self.vlm_ready:
            return "Image captioning unavailable."
        
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

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
        logger.info(f"Processing PDF: {pdf_path} (ID: {output_id})")
        
        # 1. Parse PDF
        conv_res = self.doc_converter.convert(pdf_path)
        doc = conv_res.document
        
        markdown_text = doc.export_to_markdown()
        logger.info(f"DEBUG: Markdown Length: {len(markdown_text)}")
        logger.info(f"DEBUG: Markdown Start: {markdown_text[:100]}")
        
        tables_data = []

        for i, table in enumerate(doc.tables):
            try:
                html_content = table.export_to_html()
                table_text = table.export_to_dataframe().to_string() 
                
                tables_data.append({
                    "index": i,
                    "html": html_content,
                    "summary": table_text,
                    "page": table.prov[0].page_no if table.prov else 1
                })
            except Exception as e:
                logger.warning(f"Failed to export table {i}: {e}")

        # 3. Chunking
        chunks = []
        raw_chunks = markdown_text.split("\n\n")
        logger.info(f"DEBUG: Raw chunks count: {len(raw_chunks)}")
        for i, content in enumerate(raw_chunks):
            logger.info(f"DEBUG: Chunk {i} len: {len(content.strip())} | Content: {content[:20]}...")
            if len(content.strip()) > 5:
                chunks.append({
                    "id": f"{output_id}_text_{i}",
                    "text": content,
                    "metadata": {
                        "pdf_id": output_id,
                        "source": "text",
                        "chunk_index": i
                    }
                })
        
        for i, tbl in enumerate(tables_data):
            chunks.append({
                "id": f"{output_id}_table_{i}",
                "text": f"Table Content:\n{tbl['summary']}",
                "metadata": {
                    "pdf_id": output_id,
                    "source": "table",
                    "chunk_index": i
                }
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
        record = {
            "pdf_id": output_id,
            "filename": os.path.basename(pdf_path),
            "processed_at": time.time(),
            "tables": tables_data,
            "full_markdown": markdown_text
        }
        self.metadata_db.insert(record)
        logger.info(f"Stored metadata for {output_id} in TinyDB.")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF into Knowledge Base")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output-id", required=True, help="Unique ID for this document")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM loading (images won't be captioned)")
    
    args = parser.parse_args()
    
    pipeline = IngestionPipeline(skip_vlm=args.skip_vlm)
    
    if pipeline.check_exists(args.output_id):
        logger.warning(f"Document {args.output_id} already exists. Skipping.")
        return

    try:
        pipeline.process_pdf(args.pdf, args.output_id)
        logger.info("Ingestion Complete.")
    except Exception as e:
        logger.error(f"Ingestion Failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
