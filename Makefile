.PHONY: install lint test-unit test-integration test eval

install:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/ --exclude '.venv'

test-unit:
	uv run pytest tests/unit/ -q --cov=src --cov-fail-under=85 --cov-config=pyproject.toml

test-integration:
	uv run pytest tests/integration/ -m integration -q

test: lint test-unit test-integration

eval:
	uv run python -m pipeline.evaluate
