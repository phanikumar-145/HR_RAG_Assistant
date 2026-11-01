import streamlit as st
from lab import get_collection, build_index, answer_question

# Page configuration
st.set_page_config(
    page_title="Northwind HR Assistant",
    page_icon="🤖",
    layout="centered"
)

# Application title
st.title("🤖 Northwind HR Assistant")
st.caption("Ask questions about company HR policies.")

# Load the existing RAG database
@st.cache_resource
def load_rag():
    collection = get_collection()

    if collection.count() == 0:
        build_index(collection)

    return collection

collection = load_rag()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
question = st.chat_input("Ask your HR policy question...")

if question:
    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Finding your answer..."):
            try:
                answer = answer_question(
                    collection,
                    question,
                    verbose=False
                )

                st.write(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception:
                answer = "Sorry, something went wrong. Please try again."

                st.write(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
