import streamlit as st
import pyaudio
import wave
import threading

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

def main():
    st.title("Gravador de Áudio MP3")

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

    # Botão para parar a gravação
    if st.button("Parar Gravação"):
        if st.session_state.is_recording["status"]:
            st.session_state.is_recording["status"] = False
            st.session_state.audio_thread.join()
            save_audio(st.session_state.frames, "audio.wav")
            st.success("Áudio gravado e salvo como audio.wav")

if __name__ == "__main__":
    main()
