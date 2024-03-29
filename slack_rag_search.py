import datetime

import streamlit as st
from langchain.globals import set_debug
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import dotenv
import os
from rich import print
from operator import itemgetter

from langchain import hub
from modules.slack_custom_loader import CustomSlackDirectoryLoader


from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# """
# Todo: 
# - Add specific filtering or ordering for time
# - Select specific Slack channels.
# """

# When not in env, lets' load the env file in the folder
dotenv.load_dotenv()

# Debugging 
# set_debug(True)

# This is hopefully a clean example of how Langchain can work with 
# Generating a simple retrieval for Slack.
DB_NAME = os.getenv('DB_NAME')

CONTEXT_MAX_CHARACTERS = int(os.getenv('CONTEXT_MAX_CHARACTERS'))

# This results in 32 calls to the LLM since we have a specific LLMChainExtractor
# That compresses each message so we can fit some.
AMOUNT_OF_RETRIEVED_ARTICLES = 100

# Defining the LLM (not for embedding, but for processing elements out of rag)
llm = OpenAI(temperature=0)

embeddings = OpenAIEmbeddings()
vector_db = Chroma(persist_directory=f"./store/{DB_NAME}", embedding_function=embeddings)
# vector_db = db3.similarity_search(query)

# This is a specific document_compressor chain.
compressor = LLMChainExtractor.from_llm(llm)

# This is a standard retriever that summarises what we get from the store
# compression_retriever = ContextualCompressionRetriever( base_compressor=compressor, base_retriever=retriever )

# A default output parser
output_parser = StrOutputParser()

# This is a really simple prompt, you can just print to see.
# prompt = hub.pull("rlm/rag-prompt")

prompt = ChatPromptTemplate.from_template( """
    You are an assistant analyzing chat communication of a team. Use the following messages to answer the question. 
    if asked, please extract as much possible examples with some names (and maybe dates) from the provided context.
    PLease provide quotes as to why you are reaching your conclusions.
    If you don't know the answer, just say that you don't know. \nQuestion: {question} \nContext: {context} \nAnswer:
    """ )

# https://python.langchain.com/docs/expression_language/how_to/functions

# RunnablePassthrough is what you provide through the invoke function
# I have 2 options, one that does and another that doesnt compress the output.
# Keep in mind; compressing the output takes a separate call for each doc.

chain = (
    # { "context": retriever | format_docs, "question": RunnablePassthrough() }
    prompt
    | llm
    | output_parser
)

today = datetime.datetime.now()
last_year = today.year - 1
jan_1 = datetime.date(last_year, 1, 1)
dec_31 = datetime.date(last_year, 12, 31)

result = None
docs = None 

st.header("Slack History RAG Search")
param_question = st.text_input("Ask a question", value="")

d = st.date_input(
    "With what date?",
    (jan_1, dec_31),
    jan_1,
    dec_31,
    format="MM.DD.YYYY",
)

col1, col2, col3 = st.columns(3)
with col1:
    param_minlength = st.slider('Min. characters?', 0, 1500, 300)
with col2:
    param_minreactions = st.slider('Min. responses', 0, 20, 1)
with col3:
    param_maxresults = st.slider('K Results', 0, 100, 20)

if st.button("Ask"):

    # Set it up here, so we can influence the metadata
    # https://github.com/langchain-ai/langchain/discussions/10537
    retriever = vector_db.as_retriever( 
        search_kwargs = {
            "k": param_maxresults,
            "filter": {
                '$and': [
                    {'nr_characters': {'$gt':param_minlength} },
                    {'nr_reactions': {'$gt':param_minreactions} },
                    {'date': {'$gt': 20230000000000} } # Trying to make it work with a datetime.
                ]
            }
        }
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
        
    docs = retriever.invoke( param_question )
    docs_to_string = "\n\n".join(doc.page_content for doc in docs)
    
    # Capping the string so it can be used as context
    docs_to_string_capped = docs_to_string[:CONTEXT_MAX_CHARACTERS]
    
    # https://python.langchain.com/docs/expression_language/interface
    result =  chain.invoke({ "context": docs_to_string_capped, "question": param_question } )

if docs:
    st.divider()
    st.header('Response')
    if( len(docs_to_string) > CONTEXT_MAX_CHARACTERS ):
        st.write( f"we capped the amount of characters to {CONTEXT_MAX_CHARACTERS}, original: {len(docs_to_string) }")
    st.write(result)
    st.divider()

    with st.expander("Show Documents Retrieved from the ChromaDB Store"):
        st.header('Source Documents')
        for doc in docs:
            st.write( doc.page_content )
            st.write( f"Reactions: {str(doc.metadata.get('nr_reactions'))} | Characters: {str(doc.metadata.get('nr_characters'))} | Date: {str(doc.metadata.get('date'))}" )
            st.divider()
