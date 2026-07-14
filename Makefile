.PHONY: lint typecheck test eval redteam ingest dev clean

install:
	uv sync
	uv pip install -e .

lint:
	uv run ruff check src/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/ -v

eval:
	uv run python scripts/run_eval.py

redteam:
	uv run python scripts/run_garak.py

ingest:
	uv run python scripts/ingest_pdfs.py data/pdfs/

dev:
	uv run python interfaces_app/gradio_app/app.py

clean:
	rm -rf data/chroma_db data/ingestion_ledger.db
	rm -rf data/golden_dataset/reports/*.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

.PHONY: install lint typecheck test eval redteam ingest dev clean