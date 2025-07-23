import streamlit as st
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import tempfile
import threading
import queue
import time
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import numpy as np
import av
from io import BytesIO

# Cargar variables de entorno
load_dotenv()

# Configuración de OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# Configuración de la página
st.set_page_config(
    page_title="Asistente IA Multiidioma",
    page_icon="🤖",
    layout="wide"
)

# CSS personalizado con Bootstrap
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 10px;
            animation: fadeIn 0.5s;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
        }
        .assistant-message {
            background-color: #f3e5f5;
            margin-right: 20%;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .control-panel {
            background-color: #f5f5f5;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-active {
            background-color: #4caf50;
            animation: pulse 2s infinite;
        }
        .status-inactive {
            background-color: #f44336;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

# Inicializar estados de sesión
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'es'
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = True

# Idiomas disponibles
LANGUAGES = {
    'es': {'name': 'Español', 'code': 'es-ES', 'tts': 'es'},
    'en': {'name': 'English', 'code': 'en-US', 'tts': 'en'},
    'fr': {'name': 'Français', 'code': 'fr-FR', 'tts': 'fr'},
    'de': {'name': 'Deutsch', 'code': 'de-DE', 'tts': 'de'},
    'it': {'name': 'Italiano', 'code': 'it-IT', 'tts': 'it'},
    'pt': {'name': 'Português', 'code': 'pt-PT', 'tts': 'pt'},
    'zh': {'name': '中文', 'code': 'zh-CN', 'tts': 'zh'},
    'ja': {'name': '日本語', 'code': 'ja-JP', 'tts': 'ja'},
    'ko': {'name': '한국어', 'code': 'ko-KR', 'tts': 'ko'},
    'ar': {'name': 'العربية', 'code': 'ar-SA', 'tts': 'ar'}
}

# Funciones de utilidad
def create_thread():
    """Crear un nuevo thread en OpenAI"""
    thread = client.beta.threads.create()
    return thread.id

def send_message(thread_id, message):
    """Enviar mensaje al asistente"""
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message
    )
    
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )
    
    # Esperar a que se complete la ejecución
    while run.status in ['queued', 'in_progress']:
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
    
    # Obtener mensajes
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    return messages.data[0].content[0].text.value

def text_to_speech(text, language='es'):
    """Convertir texto a voz usando gTTS"""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error en TTS: {str(e)}")
        return None

def speech_to_text(language_code='es-ES'):
    """Convertir voz a texto usando speech_recognition"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Escuchando... Habla ahora")
        r.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            text = r.recognize_google(audio, language=language_code)
            return text
        except sr.UnknownValueError:
            return "No se pudo entender el audio"
        except sr.RequestError as e:
            return f"Error en el servicio: {e}"
        except sr.WaitTimeoutError:
            return "Tiempo de espera agotado"

# Header principal
st.markdown("""
    <div class="main-header text-center">
        <h1>🤖 Asistente IA Multiidioma</h1>
        <p class="lead">Interactúa con tu asistente usando voz o texto en múltiples idiomas</p>
    </div>
    """, unsafe_allow_html=True)

# Layout principal
col1, col2 = st.columns([2, 1])

with col2:
    # Panel de control
    st.markdown("""
        <div class="control-panel">
            <h4>⚙️ Panel de Control</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Selector de idioma
    selected_lang = st.selectbox(
        "🌐 Idioma",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: f"{LANGUAGES[x]['name']}",
        index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
    )
    st.session_state.selected_language = selected_lang
    
    # Controles de voz
    st.markdown("### 🔊 Controles de Voz")
    
    col_voice1, col_voice2 = st.columns(2)
    with col_voice1:
        st.session_state.voice_enabled = st.checkbox("Habilitar voz", value=st.session_state.voice_enabled)
    
    with col_voice2:
        voice_speed = st.slider("Velocidad", 0.5, 2.0, 1.0, 0.1)
    
    # Botón de grabación
    if st.button("🎤 Grabar mensaje", use_container_width=True, type="primary"):
        with st.spinner("Grabando..."):
            text = speech_to_text(LANGUAGES[selected_lang]['code'])
            if text and not text.startswith("Error") and not text.startswith("No se pudo"):
                st.session_state.messages.append({"role": "user", "content": text})
                
                # Enviar al asistente
                if not st.session_state.thread_id:
                    st.session_state.thread_id = create_thread()
                
                response = send_message(st.session_state.thread_id, text)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Reproducir respuesta si está habilitado
                if st.session_state.voice_enabled:
                    audio_file = text_to_speech(response, LANGUAGES[selected_lang]['tts'])
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3', autoplay=True)
                        os.unlink(audio_file)
                
                st.rerun()
            else:
                st.error(text)
    
    # Estado del sistema
    st.markdown("### 📊 Estado del Sistema")
    status_html = f"""
        <div class="d-flex align-items-center">
            <span class="status-indicator status-{'active' if st.session_state.thread_id else 'inactive'}"></span>
            <span>{'Conectado' if st.session_state.thread_id else 'Desconectado'}</span>
        </div>
        """
    st.markdown(status_html, unsafe_allow_html=True)
    
    # Botón para limpiar chat
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()

with col1:
    # Área de chat
    st.markdown("### 💬 Conversación")
    
    # Contenedor de mensajes
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 Tú:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Asistente:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Input de texto
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        
        with col_input:
            user_input = st.text_input(
                "Escribe tu mensaje...",
                placeholder=f"Escribe en {LANGUAGES[selected_lang]['name']}...",
                label_visibility="collapsed"
            )
        
        with col_send:
            submitted = st.form_submit_button("📤 Enviar", use_container_width=True)
        
        if submitted and user_input:
            # Agregar mensaje del usuario
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Crear thread si no existe
            if not st.session_state.thread_id:
                st.session_state.thread_id = create_thread()
            
            # Obtener respuesta
            with st.spinner("Pensando..."):
                response = send_message(st.session_state.thread_id, user_input)
            
            # Agregar respuesta
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Reproducir respuesta si está habilitado
            if st.session_state.voice_enabled:
                audio_file = text_to_speech(response, LANGUAGES[selected_lang]['tts'])
                if audio_file:
                    st.audio(audio_file, format='audio/mp3', autoplay=True)
                    os.unlink(audio_file)
            
            st.rerun()

# Footer
st.markdown("""
    <hr>
    <div class="text-center text-muted">
        <small>Desarrollado con ❤️ usando Streamlit y OpenAI</small>
    </div>
    """, unsafe_allow_html=True)