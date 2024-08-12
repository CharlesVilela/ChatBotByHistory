import os
import time
import base64
import streamlit as st
from gtts import gTTS
from dotenv import load_dotenv
from io import BytesIO
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

from sqlalchemy import create_engine, Column, Integer, String, Text, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
from sqlalchemy import DateTime
from datetime import datetime
import pandas as pd


# Definir variáveis de ambiente para desativar avisos e resolver conflitos
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=gemini_key)

class StopCandidateException(Exception):
    pass

def _chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

# def get_vector_store(text_chunks, chunk_size=5, max_retries=5, backoff_factor=2):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss-index")
#     print('Finished vector store...')

# def get_conversational_chain():
#     try:
#         prompt_template = """
#         You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say "The answer is not available in the context". Do not provide a wrong answer.

#         Context:
#         {context}

#         Question:
#         {question}

#         Answer:
#         """
#         model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, google_api_key=gemini_key)
#         prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#         chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
#         return chain
#     except StopCandidateException as e:
#         print(f"Resposta interrompida por razões de segurança: {e}")
#         return "Desculpe, mas não posso responder a isso."
#     except Exception as e:
#         print(f"Ocorreu um erro inesperado: {e}")
#         return "Desculpe, algo deu errado."


# Função para exibir a página HTML com áudio e legendagem em tempo real
# def display_realtime_captioning(response_text):
#     audio_fp = text_to_audio(response_text)
#     audio_fp.seek(0)
#     audio_bytes = audio_fp.read()
#     audio_b64 = base64.b64encode(audio_bytes).decode()

#     # Função para exibir a página HTML com áudio e legendagem em tempo real
def display_realtime_captioning(response_text):
    audio_fp = text_to_audio(response_text)  # Suponha que esta função converte texto em um arquivo de áudio
    audio_fp.seek(0)
    audio_bytes = audio_fp.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()


    audio_html = f"""
        <audio id="audio" controls>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Seu navegador não suporta o elemento de áudio.
        </audio>

        <div id="text-container">
            <div id="text"></div>
        </div>

        <script>
            const audioElement = document.getElementById('audio');
            let recognition = new SpeechRecognition() || new webkitSpeechRecognition();
            recognition.lang = 'pt-BR'; // Idioma para o reconhecimento (português do Brasil)
            recognition.continuous = true; // Reconhecimento contínuo
            recognition.interimResults = true; // Resultados intermediários enquanto o usuário fala

            recognition.onresult = function(event) {{
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {{
                    if (event.results[i].isFinal) {{
                        transcript += event.results[i][0].transcript;
                    }} else {{
                        transcript += event.results[i][0].transcript;
                    }}
                }}
                // Atualiza as legendas em tempo real
                document.getElementById('text').innerText = transcript;
            }}

            recognition.onerror = function(event) {{
                console.error('Erro no reconhecimento de voz:', event.error);
            }}

            recognition.onend = function() {{
                console.log('Reconhecimento de voz finalizado.');
            }}

            audioElement.addEventListener('play', function() {{
                recognition.start();
            }});

            audioElement.addEventListener('pause', function() {{
                recognition.stop();
            }});
        </script>
        """

    st.markdown(audio_html,unsafe_allow_html=True)


Base = declarative_base()

# Definir a classe Message para mapear a tabela de histórico
class Message(Base):
    __tablename__ = 'history'

    id = Column(Integer, Sequence('message_id_seq'), primary_key=True)
    role = Column(String(50))
    type = Column(String(50))
    parts = Column(Text)


class UserMessage(Base):
    __tablename__ = 'user_history'

    id = Column(Integer, Sequence('usermessage_id_seq'), primary_key=True)
    question = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Criar a conexão com o banco de dados SQLite
engine = create_engine('sqlite:///chat_history.db')
new_engine = create_engine('sqlite:///user_questions.db')

# Criar a tabela no banco de dados
Base.metadata.create_all(engine)
Base.metadata.create_all(new_engine)

# Criar uma sessão para interagir com o banco de dados
Session = sessionmaker(bind=engine)
db_session = Session()

# Criar uma sessão para o novo banco de dados
NewSession = sessionmaker(bind=new_engine)
new_db_session = NewSession()


def save_user_message(question, response):
    user_message = UserMessage(question=question, response=response)
    new_db_session.add(user_message)
    new_db_session.commit()

def load_user_history():
    history = []
    messages = new_db_session.query(UserMessage).all()
    for message in messages:
        history.append({
            "question": message.question,
            "response": message.response,
            "timestamp": message.timestamp.strftime("%Y-%m-%d %H:%M:%S")  # Formatando a data e hora
        })
    return history



def save_message(role, parts):
    message = Message(role=role, parts=parts)
    db_session.add(message)
    db_session.commit()

def save_text_chunks(chunks):
    for chunk in chunks:
        text_chunk = Message(role='system', type='chunk', parts=chunk)
        db_session.add(text_chunk)
    db_session.commit()

def load_history():
    history = []
    messages = db_session.query(Message).all()
    for message in messages:
        history.append({
            "role": message.role,
            "parts": message.parts
        })
    return history

def load_text_chunks():
    chunks = db_session.query(Message).filter_by(type='chunk').all()
    return [chunk.parts for chunk in chunks]

def user_input2(user_question):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
        system_instruction="Me responda as perguntas como se fosse um professor explicando a matéria para adolescentes",
    )

    # history=[{
    #     "role": "user",
    #     "parts": [
    #         "A História do Brasil começa com a chegada dos primeiros humanos na América do Sul há pelo menos 22 000 anos AP.",
    #     ],
    # },] 
    
    print(user_question)

    # Carregar o histórico da conversa do banco de dados
    history = load_history()

    # history.append({
    #     "role": "user",
    #     "parts": [user_question]
    # })

    # Inicia a sessão de chat
    chat_session = model.start_chat()

    # Envia as mensagens do histórico
    for message in history:
        chat_session.send_message({
            "role": message["role"],
            "parts": message["parts"]
        })

    # Envia a pergunta do usuário
    response = chat_session.send_message({
        "role": "user",
        "parts": [user_question]
    })

    # # Obtém a resposta do chat
    # response = chat_session()
    # print(response.text)

    # display_realtime_captioning(response.text)

    save_user_message(user_question, response.text)
    createDataframe()

    return response

    # st.write("Reply:")
    # st.markdown(f'<div style="width: 100%; margin-top: 10px; background-color: #f9f9f9; padding: 10px; border-radius: 5px;">{response.text} 😊</div>', unsafe_allow_html=True)


def createDataframe():
    # Carregar os dados do banco de dados
    list_bd = load_user_history()

    # Verificar se há dados para evitar erro ao criar o DataFrame
    if list_bd:
        # Criar o DataFrame
        df = pd.DataFrame(list_bd)
        
        # Exportar o DataFrame para um arquivo Excel
        df.to_excel("C:\\Projetos\\chatbot4\\perguntas_e_respostas_usuario.xlsx", index=False)
    else:
        print("Nenhum dado encontrado no banco de dados.")


# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
#     new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)

#     # # Supondo a intenção da pergunta
#     # inferred_question = infer_intent(user_question)

#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()

#     # Construir histórico da conversa
#     history = "\n".join([f"User: {entry['user']}\nAI: {entry['ai']}" for entry in st.session_state.conversation_history])

#     print('Historico das pergutas: ', history)

#     response = chain(
#         {"input_documents": docs, "question": inferred_question, "history": history},
#         return_only_outputs=True
#     )

#     response_text = response["output_text"]

#     # Salvar a pergunta e a resposta no histórico
#     st.session_state.conversation_history.append({"user": user_question, "ai": response_text})

#     st.write("Reply:")
#     st.markdown(f'<div style="width: 100%; margin-top: 10px; background-color: #f9f9f9; padding: 10px; border-radius: 5px;">{response_text} 😊</div>', unsafe_allow_html=True)
    
#     audio_fp = text_to_audio(response_text)
#     audio_fp.seek(0)
#     audio_bytes = audio_fp.read()

#     audio_b64 = base64.b64encode(audio_bytes).decode()

#     audio_html = f"""
#     <audio autoplay>
#         <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
#         Seu navegador não suporta o elemento de áudio.
#     </audio>
#     """
#     st.markdown(audio_html, unsafe_allow_html=True)

def text_to_audio(text, lang='pt'):
    tts = gTTS(text=text, lang=lang)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    return audio_fp

# Função para inferir a intenção da pergunta do usuário usando BERT
# def infer_intent(user_question):
#     # Carregar o modelo e o tokenizer do BERT
#     model_name = "bert-base-uncased"
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Ajuste o número de labels conforme necessário

#     # Definir as intenções possíveis
#     intents = ["Pergunta sobre informações", "Pergunta sobre ação"]

#     # Tokenizar a pergunta do usuário
#     inputs = tokenizer(user_question, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
#     # Fazer a inferência
#     outputs = model(**inputs)
#     logits = outputs.logits
#     intent_id = torch.argmax(logits, dim=1).item()

#     inferred_intent = intents[intent_id]
#     return inferred_intent

# def test_authentication():
#     try:
#         genai.configure(api_key=gemini_key)
#         models = genai.list_models()
#         print("Autenticação bem-sucedida. Modelos disponíveis:")
#         for model in models:
#             print(model)
#     except Exception as e:
#         print(f"Erro de autenticação: {e}")
