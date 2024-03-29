import streamlit as st
from langchain.globals import set_debug
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import dotenv
import os
from rich import print

from langchain import hub
from modules.slack_custom_loader import CustomSlackDirectoryLoader

dotenv.load_dotenv()

#Name for local chromaDB, nothing needs to be set up.
DB_NAME = os.getenv('DB_NAME')

# Initially using a smaller slack export for quicker development (and less calls to openAI)
SLACK_ZIP = os.getenv('LOCAL_ZIPFILE_LARGE')

# A subset of Slack channels that you want to use for this rag application
SLACK_CHANNELS = os.getenv('CHANNELS')
channel_list = SLACK_CHANNELS.split(' ')

def load_slack_docs(channels=None, ignore_users=None) -> list[Document]:
    """Load Slack messages using a custom slack document loader"""
    slack_loader = CustomSlackDirectoryLoader( SLACK_ZIP )
    docs = slack_loader.load(channels=channels)
    return docs

# Are we sure it takes all the latest data?
# docs = load_slack_docs(channels=['general'])
docs = load_slack_docs(channel_list)

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings, persist_directory=f"./store/{DB_NAME}")