# RAG-Based Chat System 
A chat-based system implemented using RAG Architecture for Querying Fedpoffa AI-Dept HND Curriculum.

## Contact
- `Adedoyin Simeon` | [LinkedIn Profile](https://www.linkedin.com/in/adedoyin-adeyemi-a7827b160/)

## Tools
- Langchain
- Openai GPT-4o LLM
- Python
- ChromaDB (ChromaDB)
- OpenAIEmbedding
- RecursiveCharacterTextSplitter (for chunking)
- uv package manager
- Pinecone (vector db)

## Setup

- Install uv package manager
```bash
$ make install-uv
```

- Create virtual environment
```bash
$ make venv
```

- Activate the virtual environment
    - MacOs / Unix Systems (terminal)
    ```bash
    $ source .venv/bin/activate
    ```

    - Windows (Command Prompt)
    ```bash
    $ source .venv/Scripts/activate.bat
    ```
    
    - Windows (Powershel)
    ```bash
    $ source .venv/Scripts/activate.ps1
    ```

- Install dependencies
```bash
(.venv) $ make install
```

- Run app
```bash
(.venv) $ make run
```
