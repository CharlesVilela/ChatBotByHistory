# test_imports.py
import streamlit as st
from utils import text
from utils import process_embeddings

def main():
    st.set_page_config(page_title='Pergute para mim...', page_icon=':books:')

    with st.sidebar:
        st.subheader('Seus arquivos')
        pdf_docs = st.file_uploader('Carregue os seus arquivos aqui.', accept_multiple_files=True)

        if st.button('Processar'):
            all_files_text = text.process_files(pdf_docs)
            chunks = text.create_text_chunks(all_files_text)

            # print(chunks)
            
            # vectorstore = process_embeddings.create_vectorstore(chunks)
            # print(vectorstore)
            
            vectorstore2 = process_embeddings.create_vectorstoregemini(chunks)
            print(vectorstore2)


if __name__ == '__main__':

    main()
