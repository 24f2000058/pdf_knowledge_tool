# Docker Deployment Instructions

This guide explains how to build and run the Enterprise PDF Knowledge Tool using Docker.

## Prerequisites
- Docker
- Docker Compose

## Quick Start
1. **Build and Run the Application:**
   ```bash
   docker-compose up --build -d
   ```
   This command will build the image, install dependencies, and start the Streamlit service in the background on port `8501`.

2. **Access the Application:**
   Open your browser and navigate to: [http://localhost:8501](http://localhost:8501)

3. **Check Logs:**
   ```bash
   docker-compose logs -f rag-app
   ```

4. **Stop the Application:**
   ```bash
   docker-compose down
   ```

## Configuration
- Environment variables are defined in `docker-compose.yml`. You can override them by creating a `.env` file in the root directory.
- Example `.env`:
  ```ini
  LLM_BACKEND=ollama
  MODEL_NAME=llama3
  MAX_CONTEXT_TOKENS=4000
  ```

## Persistence
- Document vectors (ChromaDB), Metadata (TinyDB), and Images are stored in the `./data` directory on your host machine. This directory is mounted into the container at `/app/data`.
- This ensures your knowledge base survives container restarts.

## Ingesting Documents (CLI)
To ingest documents using the CLI within the container:
1. Ensure your PDF is in the `./data` folder (or copy it there).
2. Run the ingestion command inside the container:
   ```bash
   docker-compose exec rag-app python ingest.py --pdf data/your-document.pdf --output-id unique_doc_id
   ```
   *Note: Since `/app/data` is mounted to `./data` on your host, you can drop PDFs into your local `./data` folder and access them inside the container at `data/`.*

## Troubleshooting
- **Build Failures:** Check internet connection for package downloads. If `pip` fails, try rebuilding without cache: `docker-compose build --no-cache`.
- **Permission Issues:** If you encounter permission errors with the `data/` folder on Linux, ensure the user ID inside the container matches your host user, or `chmod` the directory appropriately.
