# RAG Tuning Log – Northwind HR Assistant

## 1. Project Overview

**Project:** Northwind HR RAG Assistant

**Objective:** Build a Retrieval-Augmented Generation system that answers employee questions using company HR policy documents.

The system uses document chunking, embeddings, vector search, and a local LLM to generate answers.

## 2. Technology Configuration

| Component | Configuration |
|---|---|
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| LLM | llama3.2:3b |
| LLM Runtime | Ollama |
| Chunk Size | 700 |
| Chunk Overlap | 120 |
| Top-K Retrieval | 4 |
| Relevance Threshold | 1.35 |
| Text Splitter | RecursiveCharacterTextSplitter |

## 3. Document Chunking

**Chunk Size:** 700 characters

**Chunk Overlap:** 120 characters

Purpose:
- Divide large policy documents into smaller chunks.
- Make relevant information easier to retrieve.
- Maintain some context between consecutive chunks.

## 4. Embedding Configuration

**Model:** all-MiniLM-L6-v2

The embedding model converts policy document chunks and user questions into numerical vectors.

These vectors are stored in ChromaDB and used for semantic similarity search.

## 5. Retrieval Configuration

**Top-K:** 4

The retriever returns the four most relevant document chunks for a user question.

**Relevance Threshold:** 1.35

A threshold is configured to reject retrieved results when their distance is above the configured limit.

Note: This is an initial configuration and should be adjusted based on further testing.

## 6. LLM Configuration

**Model:** Llama 3.2 (3B)

**Runtime:** Ollama

The LLM generates answers using the retrieved HR policy context.

The system prompt instructs the model to:
- Use only the provided policy context.
- Avoid inventing information.
- Include source document and section references.
- Respond with "I don't know based on the provided HR policies" when the answer is unsupported.

## 7. Testing Results

### Test 1: Answerable Question

**Question:** How many casual leaves are employees entitled to?

**Retrieved Information:** Leave policy document.

**Expected Answer:** 12 days of casual leave per year.

**Observed Result:** Correct answer generated with a citation to leave-policy.md, Section 2: Casual Leave.

**Status:** Passed.

### Test 2: Unsupported Question

**Question:** Does the company provide study leave?

**Observed Result:** I don't know based on the provided HR policies.

**Status:** Passed.

## 8. Observations

- The system successfully retrieves relevant HR policy information.
- Semantic search can retrieve related information even when the question uses different wording.
- The LLM generates answers using retrieved context.
- Source citations help users identify the relevant policy section.
- Unsupported questions receive a refusal response.
- Retrieved chunks may include some less relevant information, so retrieval quality can be improved.

## 9. Future Tuning Improvements

- Experiment with different chunk sizes and overlap values.
- Evaluate different embedding models.
- Improve retrieval relevance.
- Test different Top-K values.
- Implement automated evaluation metrics.
- Improve answer accuracy and response consistency.

## 10. Conclusion

The initial RAG pipeline successfully connects HR policy documents, document chunking, embeddings, ChromaDB retrieval, and local LLM answer generation.

Further tuning and evaluation can improve retrieval accuracy and overall response quality.
