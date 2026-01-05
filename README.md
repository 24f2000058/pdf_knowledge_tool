# PDF Knowledge Tool

A lightweight pipeline to convert PDFs (including scanned pages) into structured, searchable knowledge using:
- docling for PDF parsing/layout-aware extraction,
- TinyDB for document metadata and extracted table/image records,
- Chroma (local persistent) for vector embeddings and semantic search,
- transformers-based VLM (optional) for image/chart captioning,
- Streamlit app for interactive search and indexing.

What this project does
- Parses PDF structure and exports document text (docling).
- Preserves layout/TOC awareness via docling's document model.
- Splits text into section-aware chunks and stores them as vectors in Chroma.
- Extracts tables and stores HTML/JSON summaries inside TinyDB (metadata.json).
- Extracts images to data/images/ and optionally captions them using a VLM (Florence-2 via transformers).
- Provides a Streamlit search interface (search_app.py) that queries Chroma for semantic hits and shows provenance from TinyDB.
- Includes a small test suite that verifies end-to-end ingestion and stored artifacts.

Key files
- ingest.py — main ingestion pipeline
  - Uses docling.document_converter.DocumentConverter to parse PDFs.
  - Produces markdown text, extracts tables and images.
  - Stores tables and full_markdown in TinyDB (`data/metadata.json`).
  - Adds text chunks and table summaries to a Chroma collection (`data/chroma_db/`).
  - Optional VLM/image captioning via transformers (model loaded unless `--skip-vlm`).
- search_app.py — Streamlit UI
  - Connects to Chroma and TinyDB.
  - Performs semantic searches against the Chroma collection and renders results (text or table HTML).
  - Allows uploading a PDF and triggering ingest.py as a subprocess (with `--skip-vlm` toggle).
- fetch_sample.py — downloads a sample PDF to `tests/sample.pdf`.
- tests/test_pipeline.py — simple unittest validating ingest → TinyDB → Chroma behavior.

Data stores and locations
- TinyDB metadata: data/metadata.json (contains records with keys: pdf_id, filename, tables, full_markdown, processed_at).
- Chroma vector DB: data/chroma_db/ (persistent Chroma client path).
- Extracted images: data/images/
Note: .gitignore excludes `data/` so generated DBs are not committed.

Quickstart
1. Create & activate venv:
   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Fetch a sample PDF:
   python fetch_sample.py

4. Ingest a PDF:
   python ingest.py --pdf tests/sample.pdf --output-id my_doc_id
   - Add `--skip-vlm` to avoid loading the VLM (faster, no image captions).

5. Run the Streamlit search UI:
   streamlit run search_app.py

Testing
- Run tests with pytest:
  pytest -q
- tests call ingest.py and then verify TinyDB and Chroma contents.

Notes on components & behavior
- PDF parsing & OCR: docling handles layout and can fallback to OCR for scanned pages (ensure OCR dependencies are available if needed).
- Tables: extracted tables are saved into the TinyDB record for the document (as HTML and summarized text). This keeps tabular data queryable and provable via provenance fields.
- Images/Charts: saved to data/images/; optional captioning via a transformers VLM (ingest.py attempts to load `microsoft/Florence-2-base` unless `--skip-vlm`).
- Embeddings & semantic search: ingest.py creates textual chunks and table summaries and stores them in a Chroma collection named `pdf_knowledge`. search_app.py queries Chroma and uses TinyDB for provenance/details.
- Resources: loading large transformer/VLM models requires sufficient RAM/GPU and network access; use `--skip-vlm` in resource-constrained environments.

Extensibility ideas
- Replace TinyDB with MongoDB/Postgres for large-scale table storage and structured queries.
- Swap local Chroma for a managed vector DB for production scaling.
- Add language-specific OCR models or advanced denoising for low-quality scans.
- Improve table-to-schema extraction for precise numeric queries.

Troubleshooting
- If ingest.py fails to load a VLM, re-run with `--skip-vlm`.
- Chroma path is `data/chroma_db/` — delete this folder to reset the vector index.
- TinyDB is a JSON file at `data/metadata.json` — remove it to reset metadata.

License & Contributing
- Feel free to open issues or PRs. Add tests for pipeline changes and document new external services/deps in README.

