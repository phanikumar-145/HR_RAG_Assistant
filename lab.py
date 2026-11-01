import argparse
import csv
import re
from pathlib import Path

import chromadb
import ollama
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"
DB_DIR = BASE_DIR / "chroma_db"
QUESTIONS_FILE = BASE_DIR / "test_questions.csv"

COLLECTION_NAME = "hr_policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 4

# Initial threshold. We will calibrate this using retrieval distances.
RELEVANCE_THRESHOLD = 1.35

REFUSAL_MESSAGE = (
    "I don't know based on the provided HR policies."
)


# --------------------------------------------------
# 2. INITIALIZE CHROMADB
# --------------------------------------------------

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

client = chromadb.PersistentClient(path=str(DB_DIR))


def get_collection(reset=False):
    """Create or retrieve the HR policy collection."""

    if reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print("Existing collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection


# --------------------------------------------------
# 3. LOAD AND SPLIT POLICY DOCUMENTS
# --------------------------------------------------

def load_documents():
    """Read all Markdown policy documents."""

    documents = []

    for file_path in sorted(CORPUS_DIR.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        if content.strip():
            documents.append({
                "source": file_path.name,
                "content": content
            })

    if not documents:
        raise RuntimeError(
            "No policy documents found in the corpus folder."
        )

    return documents


def extract_section(text):
    """Extract the section heading from a document chunk."""

    matches = re.findall(
        r"^##\s+(.+)$",
        text,
        flags=re.MULTILINE
    )

    if matches:
        return matches[0].strip()

    return "General"


def build_index(collection):
    """Split documents and store their embeddings in ChromaDB."""

    print("\nLoading policy documents...")

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n", " ", ""]
    )

    chunks = []
    metadatas = []
    ids = []

    for document in documents:

        document_chunks = splitter.split_text(
            document["content"]
        )

        for index, chunk in enumerate(document_chunks):

            section = extract_section(chunk)

            chunks.append(chunk)

            metadatas.append({
                "source": document["source"],
                "section": section
            })

            ids.append(
                f"{document['source']}_chunk_{index}"
            )

    if chunks:
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Stored in ChromaDB: {collection.count()}")


# --------------------------------------------------
# 4. RETRIEVE RELEVANT DOCUMENT CHUNKS
# --------------------------------------------------

def retrieve_documents(collection, question, verbose=False):

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(TOP_K, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    for index, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][index]
        distance = results["distances"][0][index]

        item = {
            "document": document,
            "source": metadata["source"],
            "section": metadata["section"],
            "distance": distance
        }

        retrieved.append(item)

        if verbose:
            print("\nRetrieved chunk:")
            print("Source:", item["source"])
            print("Section:", item["section"])
            print("Distance:", round(distance, 4))
            print("Content:", document[:250], "...")

    return retrieved


# --------------------------------------------------
# 5. GENERATE GROUNDED ANSWER
# --------------------------------------------------

def answer_question(collection, question, verbose=False):

    print("\nQuestion:", question)

    retrieved = retrieve_documents(
        collection,
        question,
        verbose
    )

    # Refuse if no relevant document was retrieved.
    if not retrieved:
        print("\nAnswer:", REFUSAL_MESSAGE)
        return REFUSAL_MESSAGE

    best_distance = retrieved[0]["distance"]

    if best_distance > RELEVANCE_THRESHOLD:
        if verbose:
            print(
                "\nRefused: best retrieval distance exceeds threshold."
            )

        print("\nAnswer:", REFUSAL_MESSAGE)
        return REFUSAL_MESSAGE

    # Prepare context with source information.
    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"Source: {item['source']}\n"
            f"Section: {item['section']}\n"
            f"Content:\n{item['document']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR policy assistant for Northwind Technologies
Private Limited.

Answer the user's question using ONLY the supplied policy context.

Rules:
1. Do not use outside knowledge.
2. Do not invent policy details.
3. If the answer is not present in the context, say:
   "I don't know based on the provided HR policies."
4. Cite the source document and section for every
   policy-related answer.
5. Use this citation format:
   [source: leave-policy.md, Section 2: Casual Leave]
6. If the context does not support the answer, refuse.
7. Give a clear and concise answer.
"""

    user_prompt = f"""
POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Provide a grounded answer with citations.
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        answer = response.message.content

    except Exception as error:
        answer = f"LLM error: {error}"

    print("\nAnswer:")
    print(answer)

    return answer


# --------------------------------------------------
# 6. EVALUATE TEST QUESTIONS
# --------------------------------------------------

def evaluate_questions(collection, verbose=False):

    if not QUESTIONS_FILE.exists():
        print("test_questions.csv not found.")
        return

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        print("No test questions found.")
        return

    print("\n===== HR RAG EVALUATION =====")

    for row in rows:

        # Find the question column without depending
        # on capitalization or spaces in its heading.
        normalized = {
            key.strip().lower().replace(" ", "_"): value
            for key, value in row.items()
            if key
        }

        question = (
            normalized.get("question")
            or normalized.get("user_question")
        )

        if not question:
            print("Could not find a question column.")
            print("Available columns:", list(row.keys()))
            return

        question_id = (
            normalized.get("question_id")
            or normalized.get("id")
            or "Question"
        )

        print("\n" + "=" * 50)
        print(question_id)

        answer_question(
            collection,
            question,
            verbose
        )

    print("\n===== EVALUATION COMPLETED =====")
    print("Questions processed:", len(rows))


# --------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Northwind HR RAG Assistant"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the vector database."
    )

    parser.add_argument(
        "--ask",
        type=str,
        help="Ask an HR policy question."
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run all test questions."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display retrieval details and distances."
    )

    args = parser.parse_args()

    collection = get_collection(reset=args.reset)

    if args.reset or collection.count() == 0:
        build_index(collection)

    if args.ask:
        answer_question(
            collection,
            args.ask,
            args.verbose
        )

    elif args.evaluate:
        evaluate_questions(
            collection,
            args.verbose
        )

    elif not args.reset:
        parser.print_help()


if __name__ == "__main__":
    main()
