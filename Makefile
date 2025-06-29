# install uv
install-uv:
	pip install uv

# create a virtual environment
venv:
	uv venv

# install dependencies
install:
	uv add -r requirements.txt

# run the RAG app
run:
	python src/main.py