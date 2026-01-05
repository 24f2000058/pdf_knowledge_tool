# PDF Knowledge Tool

Small toolkit to fetch sample PDFs, ingest them into a vector store (Chroma), and run a simple search app.

## What this repository contains

- `fetch_sample.py` — helper to download or prepare sample PDF(s).
- `ingest.py` — pipeline to convert PDFs to embeddings and write them to the local Chroma DB.
- `search_app.py` — minimal app to query the ingested documents via embeddings.
- `data/` — data and DB files (generated). The Chroma DB is stored under `data/chroma_db/`.
- `tests/` — unit tests covering the pipeline.

## Quickstart

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Fetch a sample PDF (if applicable):

```bash
python fetch_sample.py
```

4. Ingest documents into the local Chroma database:

```bash
python ingest.py
```

5. Run the search app (check `search_app.py` for usage details):

```bash
python search_app.py
```

## Tests

Run the test suite with `pytest`:

```bash
pytest -q
```

## Notes & housekeeping

- The Chroma DB and generated files live in `data/chroma_db/`. These files should not be committed; `.gitignore` contains patterns to ignore them. If the DB was already committed and you want to stop tracking it, remove it from git with:

```bash
git rm --cached data/chroma_db/chroma.sqlite3
git commit -m "Stop tracking local chroma DB"
```

- If you need to reset the DB, remove `data/chroma_db/chroma.sqlite3` and re-run `ingest.py`.

## Project structure

- `fetch_sample.py`
- `ingest.py`
- `search_app.py`
- `requirements.txt`
- `data/` (generated content)
- `tests/`

## Contributing

Open an issue or submit a PR. Keep changes focused and include tests for pipeline changes.

## License

Add a license as appropriate for your project.
