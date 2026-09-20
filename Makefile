VENV= /home/hahchtar/goinfre/RAG/.venv
run:
	@uv run -m src 
install:
	mkdir -p $(VENV)
	ln -s $(VENV) .venv
	uv sync
