import streamlit as st
import os
from datetime import datetime
from gtts import gTTS
import platform
from file import initialize_qa_system, ask_question
from googletrans import Translator
from langdetect import detect

def detect_language(text):
    """
    Detect the language of the input text
    Returns language code (e.g., 'en', 'es', 'fr')
    """
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

def translate_text(text, target_lang='en'):
    """
    Translate text to target language
    """
    try:
        translator = Translator()
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        st.warning(f"Translation error: {str(e)}. Using original text.")
        return text

def text_to_speech(text, lang='en', output_file=None, slow=False):
    """
    Convert text to speech using Google Text-to-Speech (multilingual)
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join("temp", f"speech_{timestamp}.mp3")
        
        if not output_file.endswith('.mp3'):
            output_file += '.mp3'
        
        tts.save(output_file)
        return output_file
        
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None

def handle_file_upload():
    uploaded_files = st.file_uploader("Upload your documents", 
                                    type=['pdf', 'txt', 'csv', 'json'],
                                    accept_multiple_files=True)
    
    if uploaded_files:
        file_paths = []
        for file in uploaded_files:
            with open(os.path.join("temp", file.name), "wb") as f:
                f.write(file.getvalue())
            file_paths.append(os.path.join("temp", file.name))
        return file_paths
    return None

# Dictionary of language names to codes
LANGUAGES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Chinese (Mandarin)': 'zh-cn',
    'Hindi': 'hi',
    'Arabic': 'ar',
    'Bengali': 'bn',
    'Thai': 'th',
    'Turkish': 'tr',
    'Vietnamese': 'vi',
    'Tamil': 'ta'  # Added Tamil support
}

def main():
    st.set_page_config(page_title="Multilingual Document QA System", layout="wide")
    st.title("Multilingual Document Q&A System with Text-to-Speech")
    
    # Create necessary directories
    for dir_name in ["temp", "audio"]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # Initialize session state
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
        st.session_state.chat_history = []
        st.session_state.audio_files = []
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Upload Documents")
        file_paths = handle_file_upload()
        
        # Language Settings
        st.header("Language Settings")
        input_lang = st.selectbox(
            "Input Language (for questions)",
            options=list(LANGUAGES.keys()),
            index=0
        )
        output_lang = st.selectbox(
            "Output Language (for answers)",
            options=list(LANGUAGES.keys()),
            index=0
        )
        
        # TTS Settings
        st.header("Text-to-Speech Settings")
        slow_speech = st.checkbox("Slow Speech", value=False)
        auto_detect_lang = st.checkbox("Auto-detect input language", value=True)
        
        if file_paths and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                qa_chain, vector_index, context = initialize_qa_system(file_paths)
                st.session_state.qa_system = (qa_chain, vector_index, context)
                st.success("Documents processed successfully!")
    
    # Main chat interface
    if st.session_state.qa_system:
        # Question input
        question = st.text_input("Ask a question about your documents:")
        
        if question:
            # Detect or use selected input language
            detected_lang = detect_language(question) if auto_detect_lang else LANGUAGES[input_lang]
            
            # Translate question to English if needed
            if detected_lang != 'en':
                question_en = translate_text(question, target_lang='en')
            else:
                question_en = question
            
            # Get answer in English
            qa_chain, vector_index, context = st.session_state.qa_system
            answer_en = ask_question(question_en, qa_chain, vector_index, context)
            
            # Translate answer to desired output language
            output_lang_code = LANGUAGES[output_lang]
            if output_lang_code != 'en':
                answer = translate_text(answer_en, target_lang=output_lang_code)
            else:
                answer = answer_en
            
            # Generate speech for the answer
            audio_file = text_to_speech(answer, lang=output_lang_code, slow=slow_speech)
            
            # Add to chat history
            chat_entry = {
                "question": question,
                "answer": answer,
                "audio_file": audio_file,
                "detected_lang": detected_lang if auto_detect_lang else LANGUAGES[input_lang],
                "output_lang": output_lang_code
            }
            st.session_state.chat_history.append(chat_entry)
            
            if audio_file:
                st.session_state.audio_files.append(audio_file)
        
        # Display chat history with audio controls
        for chat in reversed(st.session_state.chat_history):
            col1, col2 = st.columns([1, 4])
            with col1:
                if auto_detect_lang:
                    st.write(f"*Detected language: {chat['detected_lang']}*")
            
            st.write("**Question:**", chat["question"])
            st.write("**Answer:**", chat["answer"])
            
            if chat.get("audio_file") and os.path.exists(chat["audio_file"]):
                with open(chat["audio_file"], "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")
            
            st.markdown("---")
    else:
        st.info("Please upload documents using the sidebar to start asking questions.")
    
    # Cleanup function for session end
    def cleanup():
        # Clean up temp files
        if st.session_state.qa_system and 'file_paths' in locals():
            for path in file_paths:
                if os.path.exists(path):
                    os.remove(path)
        
        # Clean up audio files
        for audio_file in st.session_state.audio_files:
            if os.path.exists(audio_file):
                os.remove(audio_file)
    
    # Register cleanup function
    st.session_state['cleanup'] = cleanup

if __name__ == "__main__":
    main()