VENV=/home/hahchtar/goinfre/VENV
TAEGET=index


server:
	@uv run -m src.api

run:
	@uv run -m src

install:
	mkdir -p $(VENV)
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync
	ln -s $(VENV) .venv

debug:
	@uv run python -m pdb -m src $(TAEGET)

lint:
	@uv run flake8  src 
	@uv run mypy --explicit-package-bases --check-untyped-defs --ignore-missing-imports src

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) -prune -exec rm -rf {} +
