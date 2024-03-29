# A RAG application on Slack messages using OpenAI, ChromaDB, Langchain and Streamlit

To use:
- Export Slack history as a ZIP file from the slack admin panel
- Copy .env.tmp to .env and change the variables to your needs
- Run `pipenv install` to install dependancies
- Log into the environment `pipenv shell`
- From within the shell load the documents into the ChromaDB: `python document_loader.py` (might take a while depending on the size of the export)
- Also from within the shell, run the streamlit app `streamlit run slack_rag_search.py`