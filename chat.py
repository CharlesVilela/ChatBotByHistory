import os
import streamlit as st
from PyPDF2 import PdfFileReader
from dotenv import load_dotenv
from utils import text
from utils import process_embeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

def main():
    print('Boa noite, mundo!')
    st.set_page_config(page_title='Pergute para mim...', page_icon=':books:')
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        process_embeddings.user_input(user_question)


    with st.sidebar:
        st.subheader('My Files')
        pdf_docs = st.file_uploader('Upload the your file...', accept_multiple_files=True)

        if st.button('Process'):
            print('Processing...')
            with st.spinner('Processing...'):
                raw_text =text.process_files(pdf_docs)
                text_chunks = text.create_text_chunks(raw_text)
                process_embeddings.get_vector_store(text_chunks)
                st.success("Done")
                # process_embeddings.test_authentication()



if __name__ == '__main__':
    main()