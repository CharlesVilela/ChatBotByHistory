import os
import speech_recognition as sr
import streamlit as st
import pyaudio
import wave
import threading
from PyPDF2 import PdfFileReader
from dotenv import load_dotenv
from utils import text as t
from utils import process_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

# Função para gravar áudio
def record_audio(frames, is_recording):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    while is_recording["status"]:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

# Função para salvar o áudio
def save_audio(frames, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()

# Função para converter áudio em texto
def audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='pt-BR')
        except sr.UnknownValueError:
            text = "Não foi possível entender o áudio."
        except sr.RequestError as e:
            text = f"Erro na requisição ao serviço de reconhecimento de fala: {e}"
    return text

def main():
    st.set_page_config(page_title='Pergunte para mim...', page_icon=':books:')
    st.header("ChronoChat")

    # Adicionando estilo CSS para ajustar a altura dos botões
    st.markdown("""
        <style>
        .stButton > button {
            height: 50px;
            margin-top: 28px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Inicializar estados
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = {"status": False}
    if "frames" not in st.session_state:
        st.session_state.frames = []
    if "audio_thread" not in st.session_state:
        st.session_state.audio_thread = None
    if "recognized_text" not in st.session_state:
        st.session_state.recognized_text = ""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Criar colunas para organizar o layout
    col1, col2 = st.columns([0.75, 0.25])

    list_text = []

    with col1:
        user_question = st.text_input("Ask a Question from the PDF Files")
        # Exibir texto reconhecido, se disponível
        if st.session_state.recognized_text:
            st.text_area("Texto Reconhecido", st.session_state.recognized_text, height=150)
        if user_question:
            list_text.append(user_question)
    with col2:
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            if st.button("Iniciar Gravação"):
                if not st.session_state.is_recording["status"]:
                    st.session_state.is_recording = {"status": True}
                    st.session_state.frames = []
                    st.session_state.audio_thread = threading.Thread(target=record_audio, args=(st.session_state.frames, st.session_state.is_recording))
                    st.session_state.audio_thread.start()
                    st.warning("Gravando áudio...")

        with col2_2:
            if st.button("Parar Gravação"):
                if st.session_state.is_recording["status"]:
                    st.session_state.is_recording["status"] = False
                    st.session_state.audio_thread.join()
                    save_audio(st.session_state.frames, "audio.wav")
                    st.success("Áudio gravado e salvo como audio.wav")

                    # Converter áudio em texto
                    text = audio_to_text("audio.wav")
                    st.session_state.recognized_text = text

                    # Atualizar pergunta do usuário com o texto reconhecido
                    if text:
                        list_text.append(text)
    
    for value in list_text:
        if value is not None and value != "":
            with st.spinner("Gerando resposta..."):
                process_embeddings.user_input2(value)
                # st.write(response)
            list_text.clear()

    # with st.sidebar:
    #     st.subheader('My Files')
    #     pdf_docs = st.file_uploader('Upload the your file...', accept_multiple_files=True)
    #     if st.button('Process'):
    #         with st.spinner('Processing...'):
    #             raw_text = t.process_files(pdf_docs)
    #             text_chunks = t.create_text_chunks(raw_text)
    #             process_embeddings.get_vector_store(text_chunks)
    #             st.success("Done")

if __name__ == '__main__':
    main()
