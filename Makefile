.PHONY: install lint fmt test cov dev wecom-worker db-up db-down db-migrate clean

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check . --fix
	uv run ruff format .

test:
	uv run pytest

cov:
	uv run pytest --cov=backend --cov-report=term-missing --cov-fail-under=80

dev:
	uv run fastapi dev backend/app/main.py

wecom-worker:
	uv run python -m backend.app.wecom_aibot_worker

db-up:
	docker compose -f docker/docker-compose.yml up -d postgres redis etcd minio milvus

db-down:
	docker compose -f docker/docker-compose.yml down

db-migrate:
	bash scripts/db_apply.sh

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} +
