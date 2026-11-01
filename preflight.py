
import sys
import importlib.util
from pathlib import Path
import ollama

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"

print("\n===== HR RAG ASSISTANT PREFLIGHT CHECK =====")

# 1. Python version
print("\n1. Python Version:", sys.version)

# 2. Required packages
packages = [
    "chromadb",
    "sentence_transformers",
    "langchain_text_splitters",
    "ollama"
]

print("\n2. Checking packages:")
for package in packages:
    if importlib.util.find_spec(package):
        print(f"   PASS: {package}")
    else:
        print(f"   FAIL: {package}")

# 3. Policy documents
print("\n3. Checking policy documents:")
documents = [
    "leave-policy.md",
    "expense-policy.md",
    "remote-work-policy.md"
]

for document in documents:
    path = CORPUS_DIR / document
    if path.exists() and path.stat().st_size > 0:
        print(f"   PASS: {document}")
    else:
        print(f"   FAIL: {document}")

# 4. Ollama model
print("\n4. Checking Ollama model:")
try:
    models = ollama.list()
    model_names = [
        model.model for model in models.models
    ]

    if any("llama3.2:3b" in name for name in model_names):
        print("   PASS: llama3.2:3b is available")
    else:
        print("   FAIL: llama3.2:3b not found")

except Exception as error:
    print("   FAIL: Ollama connection error:", error)

# 5. LLM response
print("\n5. Testing LLM response:")
try:
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": "Reply with only the word WORKING."
            }
        ]
    )

    print("   PASS:", response.message.content)

except Exception as error:
    print("   FAIL:", error)

print("\n===== PREFLIGHT CHECK COMPLETED =====")
