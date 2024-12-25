import streamlit as st
import os
from file import initialize_qa_system, ask_question  # Assuming previous code is in backend.py

st.set_page_config(page_title="Document QA System", layout="wide")

def handle_file_upload():
    uploaded_files = st.file_uploader("Upload your documents", 
                                    type=['pdf', 'txt', 'csv', 'json'],
                                    accept_multiple_files=True)
    
    if uploaded_files:
        file_paths = []
        for file in uploaded_files:
            # Save uploaded file temporarily
            with open(os.path.join("temp", file.name), "wb") as f:
                f.write(file.getvalue())
            file_paths.append(os.path.join("temp", file.name))
        return file_paths
    return None

def main():
    st.title("Document Q&A System")
    
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Initialize session state
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
        st.session_state.chat_history = []
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Documents")
        file_paths = handle_file_upload()
        
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
            qa_chain, vector_index, context = st.session_state.qa_system
            answer = ask_question(question, qa_chain, vector_index, context)
            
            # Add to chat history
            st.session_state.chat_history.append({"question": question, "answer": answer})
        
        # Display chat history
        for chat in reversed(st.session_state.chat_history):
            st.write("**Question:**", chat["question"])
            st.write("**Answer:**", chat["answer"])
            st.markdown("---")
    else:
        st.info("Please upload documents using the sidebar to start asking questions.")
    
    # Cleanup temp files when session ends
    if st.session_state.qa_system:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    main()