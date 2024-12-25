import warnings
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import PyPDF2
import csv
import google.generativeai as genai
from tkinter import filedialog
import tkinter as tk

warnings.filterwarnings("ignore")
global context

def get_file_inputs():
    """
    Opens a file dialog to allow users to select multiple files.
    Returns a list of file paths.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(
        title='Select files',
        filetypes=[
            ('All Supported Files', '*.pdf;*.txt;*.csv;*.json'),
            ('PDF Files', '*.pdf'),
            ('Text Files', '*.txt'),
            ('CSV Files', '*.csv'),
            ('JSON Files', '*.json')
        ]
    )
    return list(file_paths)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding='latin-1') as f:
            return f.read()

def extract_text_from_json(json_path):
    with open(json_path, "r", encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not data:  
                return ""
            return json.dumps(data, indent=4)
        except json.JSONDecodeError:
            return ""

def read_and_structure_csv(csv_path):
    structured_data = []
    with open(csv_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if 'plan_type' in row:
                plan_details = f"plan_type: {row['plan_type']}\n"
                for key, value in row.items():
                    if key != 'plan_type':
                        plan_details += f"  - **{key.replace('_', ' ').title()}**: {value}\n"
            else:
                plan_details = "\n".join(f"{key}: {value}" for key, value in row.items())
            structured_data.append(plan_details)
    return "\n\n".join(structured_data)

def initialize_qa_system(file_paths):
    """
    Initialize the QA system with the given file paths
    """
    texts1 = []
    for path in file_paths:
        try:
            if path.lower().endswith(".pdf"):
                texts1.append(extract_text_from_pdf(path))
            elif path.lower().endswith(".txt"):
                texts1.append(extract_text_from_txt(path))
            elif path.lower().endswith(".csv"):
                texts1.append(read_and_structure_csv(path))
            elif path.lower().endswith(".json"):
                texts1.append(extract_text_from_json(path))
            print(f"Successfully processed: {path}")
        except Exception as e:
            print(f"Error processing file {path}: {str(e)}")

    context = "\n\n".join(texts1)

    # Initialize text splitter and vector index
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=11000, chunk_overlap=1000)
    texts = text_splitter.split_text(context)

    api_key = "AIzaSyASchPTlryLW0x-X8JaS59U1EztfIPeixk"
    if not api_key:
        raise ValueError("API key not found. Please set your GEMINI_API_KEY in the environment.")

    model = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=api_key,
        temperature=0.1, 
        convert_system_message_to_human=True
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

    # Create QA chain
    template = """Understand the contents in the document and answer the questions
    Context:
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain, vector_index, context

def load_history():
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):  
                    return data
            except json.JSONDecodeError:
                pass  
    return []

def ask_question(question, qa_chain, vector_index, context):
    if question.strip().lower() == "exit":
        return "Goodbye! Feel free to start a new session anytime."

    with open("./chat_history.txt", "a") as f:
        f.write(f"USER: {question}\n")

    result = qa_chain({"query": question})
    answer = result["result"]
    return answer

def main():
    print("Please select the files you want to analyze...")
    file_paths = get_file_inputs()
    
    if not file_paths:
        print("No files selected. Exiting...")
        return

    print("\nInitializing QA system with selected files...")
    qa_chain, vector_index, context = initialize_qa_system(file_paths)
    
    print("\nSystem ready! You can now ask questions about the documents.")
    print("Type 'exit' to quit or 'reload' to select different files.")
    
    while True:
        question = input("\nAsk your question: ")
        
        if question.strip().lower() == 'exit':
            print("Goodbye!")
            break
        elif question.strip().lower() == 'reload':
            print("\nPlease select new files...")
            file_paths = get_file_inputs()
            if file_paths:
                print("Reinitializing system with new files...")
                qa_chain, vector_index, context = initialize_qa_system(file_paths)
                print("System ready with new files!")
            continue
            
        answer = ask_question(question, qa_chain, vector_index, context)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()