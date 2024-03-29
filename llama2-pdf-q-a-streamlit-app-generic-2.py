import json
import os
import streamlit as st
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext

from llama_index.vector_stores import CassandraVectorStore
from copy import deepcopy
from tempfile import NamedTemporaryFile
from fine_tune import FineTuner

@st.cache_resource
def create_datastax_connection():

    cloud_config= {'secure_connect_bundle': 'secure-connect-pdf-summarization.zip'}

    with open("pdf-summarization-token.json") as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_session = cluster.connect()
    return astra_session

def main():

    index_placeholder = None
    st.set_page_config(page_title = "Chat with your PDF using Llama2 & Llama Index", page_icon="🦙")
    st.header('🦙 Chat with your PDF using Llama2 model & Llama Index')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

    session = create_datastax_connection()

    os.environ['GRADIENT_ACCESS_TOKEN'] = "49SxShZ6rRvQu8YVAeSjoj3a91rqBWZm"
    os.environ['GRADIENT_WORKSPACE_ID'] = "6114a445-d716-4ff4-ac7b-a7ab9ad42995_workspace"

    
    

    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF & Click to process',
                                accept_multiple_files = False, 
                                type=['pdf'])
        if st.button('Process'):
            with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
                f.write(docs.getbuffer())
                with st.spinner('Processing'):
                    fine_tuner = FineTuner(model_name = "FineTunedLlama2", num_epochs = 5)
                    service_context = fine_tuner.fine_tune()
                    documents = SimpleDirectoryReader(".").load_data()
                    index = VectorStoreIndex.from_documents(documents,
                                                            service_context=service_context)
                    query_engine = index.as_query_engine()
                    if "query_engine" not in st.session_state:
                        st.session_state.query_engine = query_engine
                    st.session_state.activate_chat = True

    if st.session_state.activate_chat == True:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", 
                                              "avatar" :'👨🏻',
                                              "content": prompt})

            query_index_placeholder = st.session_state.query_engine
            pdf_response = query_index_placeholder.query(prompt)
            #cleaned_response = pdf_response.response
            cleaned_response = pdf_response
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(pdf_response)
            st.session_state.messages.append({"role": "assistant", 
                                              "avatar" :'🤖',
                                              "content": pdf_response})
        else:
            st.markdown(
                'Upload your PDFs to chat'
                )


if __name__ == '__main__':
    main()
