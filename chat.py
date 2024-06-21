import os
import speech_recognition as sr
import numpy as np
import av
import streamlit as st
import pyaudio
import wave
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
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

# Definindo a classe AudioProcessor fora da função main
# class AudioProcessor(AudioProcessorBase):
    # def __init__(self):
    #     super().__init__()
    #     self.recorder = sr.Recognizer()
    #     self.audio_buffer = []
    #     self.text = ""

    # def recv_queued(self, frames: list[av.AudioFrame]) -> None:
    #     # Adicionar frames ao buffer
    #     for frame in frames:
    #         audio = frame.to_ndarray()
    #         self.audio_buffer.append(audio)

    #     # Processar áudio no buffer se houver dados suficientes
    #     if len(self.audio_buffer) > 0:
    #         audio_data = np.concatenate(self.audio_buffer, axis=0)
    #         audio_data = sr.AudioData(audio_data.tobytes(), sample_rate=frames[0].sample_rate, sample_width=frames[0].format.sample_size)
    #         try:
    #             self.text = self.recorder.recognize_google(audio_data, language="pt-BR")
    #             print('Texto convertido: ', self.text)
    #         except sr.UnknownValueError:
    #             self.text = "Não foi possível reconhecer o áudio"
    #             print(self.text)
    #         except sr.RequestError as e:
    #             self.text = f"Erro no serviço de reconhecimento: {e}"
    #             print(self.text)
            
    #         # Limpar buffer após processamento
    #         self.audio_buffer = []

    # def get_text(self):
    #     return self.text


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
    st.header("Chat with PDF using Gemini")
    st.title("Pergunte usando a sua voz")
    user_question = st.text_input("Ask a Question from the PDF Files")

    # # WebRTC para gravação de áudio
    # webrtc_ctx = webrtc_streamer(
    #     key="example", 
    #     mode=WebRtcMode.SENDONLY,
    #     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    #     media_stream_constraints={"audio": True, "video": False},
    #     audio_processor_factory=AudioProcessor,
    #     async_processing=True,
    # )

    # if webrtc_ctx.audio_processor:
    #     recognized_text = webrtc_ctx.audio_processor.get_text()
    #     if recognized_text:
    #         st.text_input("Pergunta reconhecida:", value=recognized_text)


    st.title("Gravador de audio MP3")

    # Inicializar estados
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = {"status": False}
    if "frames" not in st.session_state:
        st.session_state.frames = []
    if "audio_thread" not in st.session_state:
        st.session_state.audio_thread = None

    # Botão para iniciar a gravação
    if st.button("Iniciar Gravação"):
        if not st.session_state.is_recording["status"]:
            st.session_state.is_recording["status"] = True
            st.session_state.frames = []
            st.session_state.audio_thread = threading.Thread(target=record_audio, args=(st.session_state.frames, st.session_state.is_recording))
            st.session_state.audio_thread.start()
            st.warning("Gravando áudio...")

    text = ""
    # Botão para parar a gravação
    if st.button("Parar Gravação"):
        if st.session_state.is_recording["status"]:
            st.session_state.is_recording["status"] = False
            st.session_state.audio_thread.join()
            save_audio(st.session_state.frames, "audio.wav")
            st.success("Áudio gravado e salvo como audio.wav")

            # Converter áudio em texto
            text = audio_to_text("audio.wav")
            st.text_area("Texto Reconhecido", text)


    print("Mostrando o audio convertido: ", text)

    if text is not None:
        user_question = text
        print("Mostrando o user question: ",user_question)

    if user_question:
        process_embeddings.user_input(user_question)

    with st.sidebar:
        st.subheader('My Files')
        pdf_docs = st.file_uploader('Upload the your file...', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing...'):
                raw_text = text.process_files(pdf_docs)
                text_chunks = text.create_text_chunks(raw_text)
                process_embeddings.get_vector_store(text_chunks)
                st.success("Done")

if __name__ == '__main__':
    main()
