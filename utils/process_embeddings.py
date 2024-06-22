import os
import time
import base64

import streamlit as st
from gtts import gTTS

from dotenv import load_dotenv
from io import BytesIO

from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=gemini_key)

# def create_vectorstore(chunks):
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
#     vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
#     return vectorstore

def _chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

def get_vector_store(text_chunks, chunk_size=5, max_retries=5, backoff_factor=2):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
    # all_embeddings = []
    # for chunk in _chunks(text_chunks, chunk_size):
    #     retry_count = 0
    #     while retry_count < max_retries:
    #         try:
    #             print(f"Tentando embutir documentos. Tentativa {retry_count + 1} de {max_retries}")
    #             embeddings_chunk = embeddings.embed_documents(chunk)
    #             all_embeddings.extend(embeddings_chunk)
    #             break  # Sucesso, sai do loop de retry
    #         except Exception as e:
    #             print(f"Erro na tentativa {retry_count + 1}: {e}")
    #             retry_count += 1
    #             if retry_count >= max_retries:
    #                 raise Exception(f"Falha ao embutir documentos após {max_retries} tentativas: {e}")
    #             time.sleep(backoff_factor ** retry_count)  # Exponential backoff

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    print("Mostrando o vector store...")
    print(vector_store)
    vector_store.save_local("faiss-index")
    print('Finished vector store...')

def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n {question}\n

    Answer:

    """

    model=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3,google_api_key=gemini_key)

    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model,chain_type="stuff", prompt=prompt)
    
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)

    new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    # st.write("Reply: ", response["output_text"])
    st.write("Reply:")
    st.markdown(f'<div style="width: 100%; margin-top: 10px;">{response["output_text"]}</div>', unsafe_allow_html=True)
    
    
    audio_fp=text_to_audio(response["output_text"])
    # audio_file = open("output.mp3", "rb")
    # audio_bytes = audio_file.read()
    # st.audio(audio_bytes, format='audio/mp3').
    audio_fp.seek(0)
    audio_bytes = audio_fp.read()

    audio_b64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        Seu navegador não suporta o elemento de áudio.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)



def text_to_audio(text, lang='pt'):
    tts = gTTS(text=text, lang=lang)
    # tts.save('output.mp3')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    return audio_fp



def test_authentication():
    try:
        genai.configure(api_key=gemini_key)
        models = genai.list_models()
        print("Autenticação bem-sucedida. Modelos disponíveis:")
        for model in models:
            print(model)
    except Exception as e:
        print(f"Erro de autenticação: {e}")
