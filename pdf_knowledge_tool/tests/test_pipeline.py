
import unittest
import os
import shutil
from tinydb import TinyDB, Query
import chromadb
from pathlib import Path

import time

# Config
TEST_ID = f"test_doc_{int(time.time())}"
PDF_PATH = "tests/sample.pdf"
DB_DIR = "tests/data"

class TestPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Clean up previous test runs
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)
        
        # We need to make sure ingest.py uses these paths. 
        # Since ingest.py has hardcoded paths, for this test we might need to modify it 
        # or just run it and check the default ./data folder. 
        # For simplicity in this env, let's rely on the default ./data and clean it up or ignore.
        # ALLOWING default ./data so we test the actual script behavior.
        pass

    def test_01_ingestion(self):
        print(f"\nRunning Ingestion on {PDF_PATH}...")
        exit_code = os.system(f"py -3.12 ingest.py --pdf {PDF_PATH} --output-id {TEST_ID} --skip-vlm")
        self.assertEqual(exit_code, 0, "Ingestion script failed")

    def test_02_tinydb_verification(self):
        print("\nVerifying TinyDB...")
        db = TinyDB('./data/metadata.json')
        doc = db.get(Query().pdf_id == TEST_ID)
        self.assertIsNotNone(doc, "Document not found in TinyDB")
        self.assertEqual(doc['filename'], 'sample.pdf')
        print(f"TinyDB Record: {doc.keys()}")

    def test_03_chroma_verification(self):
        print("\nVerifying ChromaDB...")
        client = chromadb.PersistentClient(path="./data/chroma_db")
        collection = client.get_collection("pdf_knowledge")
        
        # Query by metadata
        results = collection.get(where={"pdf_id": TEST_ID})
        count = len(results['ids'])
        print(f"Chroma Chunks Found: {count}")
        self.assertGreater(count, 0, "No chunks found in ChromaDB for test doc")

if __name__ == '__main__':
    unittest.main()
