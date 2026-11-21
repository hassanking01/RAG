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