# Northwind HR RAG Assistant

An AI-powered HR Policy Assistant that answers employee questions using company policy documents through Retrieval-Augmented Generation (RAG).

## Project Overview

The Northwind HR RAG Assistant helps employees find information from HR policy documents quickly and easily.

Instead of manually searching through policy documents, users can ask questions in natural language and receive answers generated using the relevant policy information.

The application uses a local Large Language Model (LLM) through Ollama and a Streamlit-based chat interface.

## Key Features

- **Retrieval-Augmented Generation (RAG):** Retrieves relevant information from HR policy documents.
- **Document Chunking:** Splits policy documents into smaller chunks for retrieval.
- **Embeddings:** Converts document chunks into numerical representations.
- **Vector Database:** Uses ChromaDB to store and retrieve document embeddings.
- **Semantic Search:** Finds relevant policy information based on user questions.
- **Local LLM:** Uses Llama 3.2 (3B) through Ollama to generate answers.
- **Source Citations:** Provides policy document names and section references.
- **Hallucination Control:** Responds with "I don't know based on the provided HR policies" when the information is not available in the retrieved context.
- **Streamlit Interface:** Provides a simple interactive chatbot.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| Ollama | Local LLM execution |
| Llama 3.2 (3B) | Answer generation |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| LangChain Text Splitters | Document chunking |

## Project Structure

```text
HR_RAG_Assistant/
│
├── corpus/
│   ├── leave-policy.md
│   ├── expense-policy.md
│   └── remote-work-policy.md
│
├── app.py
├── lab.py
├── preflight.py
├── requirements.txt
├── test_questions.csv
├── tuning-log.md
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/HR_RAG_Assistant.git
cd HR_RAG_Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Download Ollama from:

https://ollama.com/

Download the required model:

```bash
ollama pull llama3.2:3b
```

### 5. Build the RAG index

```bash
python lab.py --reset
```

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## Testing

Run the individual question test:

```bash
python lab.py --ask "How many casual leaves are employees entitled to?" --verbose
```

Run the evaluation questions:

```bash
python lab.py --evaluate
```

The test dataset contains 12 questions:
- 9 questions answerable from the policy documents.
- 3 questions that should receive an unsupported-answer response.

## HR Policy Documents

The application currently uses three policy documents:

1. Leave Policy
2. Expense Reimbursement Policy
3. Remote Work Policy

## Important Notes

- Ollama must be running before using the application.
- The ChromaDB database is generated locally when the documents are indexed.
- The embedding model may need to be downloaded during the first run.
- Answers are restricted to the available HR policy documents.
- The policies are fictional documents created for this project.

## Future Improvements

- Improve retrieval accuracy.
- Add more HR policy documents.
- Implement automated evaluation metrics.
- Improve the chatbot interface.
- Add document upload functionality.

## Author

Developed as an AI/ML project using Retrieval-Augmented Generation, vector databases, local LLMs, and Streamlit.
