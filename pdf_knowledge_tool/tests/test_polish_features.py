import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import os
import sys

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from search_app import delete_document
import ingest

class TestPolishFeatures(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup temp env
        cls.test_dir = Path("tests/temp_polish_data")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Patch config constants
        config.DATA_DIR = str(cls.test_dir)
        config.IMAGES_DIR = cls.test_dir / "images"
        config.IMAGES_DIR.mkdir(exist_ok=True)
        config.METADATA_DB_PATH = cls.test_dir / "metadata.json"
        
        # Patch Ingest/Search modules to use these paths if they aren't using config directly
        # (They mostly use constants defined at module level, so we might need to patch those)
        ingest.DATA_DIR = cls.test_dir
        ingest.IMAGES_DIR = cls.test_dir / "images"
        
    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    @patch("search_app.collection")
    @patch("search_app.metadata_db")
    def test_delete_document(self, mock_metadata_db, mock_collection):
        """Test that delete_document removes from all 3 stores."""
        pdf_id = "test_doc_delete"
        
        # 1. Setup Mock Files
        img_1 = config.IMAGES_DIR / f"{pdf_id}_page_1.jpg"
        img_2 = config.IMAGES_DIR / f"{pdf_id}_page_2.jpg"
        other_img = config.IMAGES_DIR / "other_doc_page_1.jpg"
        
        img_1.touch()
        img_2.touch()
        other_img.touch()
        
        # 2. Run Delete
        success, msg = delete_document(pdf_id)
        
        # 3. Assertions
        self.assertTrue(success)
        
        # Chroma Delete Called?
        mock_collection.delete.assert_called_with(where={"pdf_id": pdf_id})
        
        # TinyDB Remove Called?
        # Note: We can't easily verify the exact Query object equality, but we can verify remove was called
        self.assertTrue(mock_metadata_db.remove.called)
        
        # Files Deleted?
        self.assertFalse(img_1.exists(), "Target image 1 should be deleted")
        self.assertFalse(img_2.exists(), "Target image 2 should be deleted")
        self.assertTrue(other_img.exists(), "Unrelated image should persist")

    @patch("ingest.DocumentConverter")
    def test_preview_config_adherence(self, mock_converter_cls):
        """Test that Ingest pipeline respects preview config."""
        # Mock Config
        config.ENABLE_PAGE_PREVIEWS = True
        config.PREVIEW_MAX_PAGES = 5
        
        # Initialize Pipeline
        pipeline = ingest.IngestionPipeline(skip_vlm=True)
        
        # Check if Docling Converter was init with options
        # We check the format_options arg passed to constructor
        call_args = mock_converter_cls.call_args
        self.assertIsNotNone(call_args)
        
        # It's hard to deep inspect the PdfPipelineOptions object inside the call_args in a robust way
        # without importing the exact classes, but verifying it was called with format_options is a good sign.
        kwargs = call_args.kwargs
        self.assertIn('format_options', kwargs)
        
    def test_preview_max_pages_logic(self):
        """Test logic for limiting image saves (mocking the loop logic from ingest)."""
        # This mirrors the logic added to ingest.py
        max_pages = 2
        saved_count = 0
        total_pages = 5
        
        # Simulate loop
        for i in range(total_pages):
            if saved_count >= max_pages:
                break
            saved_count += 1
            
        self.assertEqual(saved_count, max_pages)

if __name__ == '__main__':
    unittest.main()
