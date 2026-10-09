VENV=/home/hahchtar/goinfre/VENV

run:
	@uv run -m src

install:
	mkdir -p $(VENV)
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync
	ln -s $(VENV) .venv
