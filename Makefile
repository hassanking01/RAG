VENV= /home/hahchtar/goinfre/RAG/
run:
	@uv run -m src 
install:
	
	mkdir -p $(VENV)
	uv venv "$(VENV)/.venv"
	ln -s "$(VENV)/.venv" .venv
	uv sync
