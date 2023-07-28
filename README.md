# AMSearch

Semantic text search using AM Stemmer and BERT-based embeddings.

## Deployment

You'll need a pre-trained model for TF-IDF and BERT before running the web. The model can be downloaded from Azure Blob Storage.

1. Clone this repo.
2. Copy model blobs to a directory, default to `./data`
3. Create env file, `postgres.env` and `web.env` from their respective example files
4. Run `docker compose up`

You should be able to access the web at http://localhost:8000. The database migrations will automatically run at first startup.
