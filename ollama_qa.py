import os
import streamlit as st
import sentry_sdk
import sentry_sdk.profiler

from dotenv import load_dotenv
from ollama_rag import RAGApp
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

sentry_sdk.init(
    dsn="https://928cdfed3291d120bd9972df53e4d90d@o4505835180130304.ingest.us.sentry.io/4508009613164544",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

# Constants
DOCS_FOLDER = os.getenv("DOCS_FOLDER")
DEFAULT_DOCUMENT = os.getenv(
    "DEFAULT_DOCUMENT"
)  # Default document to load when nothing is uploaded

rag_app = RAGApp()

# Streamlit Configuration
st.set_page_config(page_title="Q&A Chat", layout="centered")


# Initialize Session State
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        rag_app.prime_llama_in_background()


# Handle Document Upload
def handle_document_upload(uploaded_file):
    if uploaded_file:
        file_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None


def main():
    initialize_session_state()

    st.title("Q&A Chat")
    st.write("Ask a question")

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    question = st.chat_input("Question")

    if question is not None and question != "":

        with st.chat_message("user"):
            st.markdown(question)
            st.session_state.chat_history.append(HumanMessage(content=question))

        with st.chat_message("assistant"):
            sentry_sdk.profiler.start_profiler()

            with sentry_sdk.start_transaction(op="task", name="Front end processing"):
                with sentry_sdk.start_span(description="Answer question"):
                    ai_response = rag_app.run_retrieval_chain(
                        question=question,
                        chat_history=st.session_state.chat_history,
                    )

            # st.write_stream(ai_response)
            response_container = st.empty()

            final_response = ""

            for chunk in ai_response:
                final_response += chunk.replace("$", "\$")
                response_container.markdown(final_response)

            st.session_state.chat_history.append(AIMessage(content=final_response))

            sentry_sdk.profiler.stop_profiler()


if __name__ == "__main__":
    main()
