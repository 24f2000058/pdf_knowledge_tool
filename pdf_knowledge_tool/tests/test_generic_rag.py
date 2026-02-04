import unittest
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import logging

# Adjust path to import modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import LLMService, classify_answer_style
from ingest import IngestionPipeline, DocumentMetadata
from schemas import ChunkMetadata
import config

class TestGenericRAG(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup temp data dir
        cls.test_dir = Path("tests/temp_generic_data")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Patch config
        config.DATA_DIR = str(cls.test_dir)
        config.METADATA_DB_PATH = cls.test_dir / "metadata.json"
        
        # Patch ingest module constants
        import ingest
        ingest.DATA_DIR = cls.test_dir
        ingest.METADATA_DB_PATH = cls.test_dir / "metadata.json"
        ingest.CHROMA_DB_PATH = cls.test_dir / "chroma_db"
        ingest.IMAGES_DIR = cls.test_dir / "images"
        ingest.COLLECTION_NAME = "test_collection" # Ensure no collision
        
    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_answer_style_heuristic(self):
        """Test the generic answer typing logic."""
        self.assertEqual(classify_answer_style("What are the requirements?"), "list")
        self.assertEqual(classify_answer_style("List the steps to apply."), "list")
        self.assertEqual(classify_answer_style("Why is the sky blue?"), "explanation")
        self.assertEqual(classify_answer_style("What is the capital of France?"), "short_factual")

    @patch('models.LLMService._initialize')
    @patch('models.LLMService.generate_answer_with_retry')
    def test_prompt_leakage_prevention(self, mock_generate, mock_init):
        """Mock test to ensure prompt isn't in output (logic check via code review mostly, but we trigger the flow)."""
        # Reset singleton
        LLMService._instance = None
        
        llm = LLMService()
        # Mock backend to transformers to trigger the transformers logic path if we were running real code,
        # but here we mock generate_answer_with_retry anyway.
        llm.backend = "transformers"
        llm.count_tokens = MagicMock(return_value=10) # Mock token counter
        
        # Mock payload: Suppose model returns "Answer: The answer."
        mock_generate.return_value = "The critical requirements are X and Y."
        
        context = "The requirement is X and Y."
        question = "What are the requirements?"
        
        answer = llm.generate_answer(context, question)
        
        # Assert generic prompt parts are NOT in answer
        self.assertNotIn("You are a strict extraction assistant", answer)
        self.assertNotIn("User Question:", answer)
        self.assertEqual(answer, "The critical requirements are X and Y.")

    def test_duplicate_ingestion(self):
        """Verify hash-based deduplication logs/skips."""
        # Create dummy PDF
        pdf_path = self.test_dir / "dummy.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b"dummy content")
            
        pipeline = IngestionPipeline(skip_vlm=True)
        
        # Override process_pdf internals or just check hash logic
        # Ideally we run process_pdf, but dependency on docling might be heavy for unit test
        # Let's test the hash check logic directly.
        
        file_hash = pipeline.calculate_file_hash(str(pdf_path))
        
        # 1. First Pass: Not exists
        self.assertFalse(pipeline.check_exists(file_hash))
        
        # 2. Insert Mock Metadata
        pipeline.metadata_db.insert({"file_hash": file_hash, "id": "test_1"})
        
        # 3. Second Pass: Should exist
        self.assertTrue(pipeline.check_exists(file_hash))
        
if __name__ == '__main__':
    unittest.main()
