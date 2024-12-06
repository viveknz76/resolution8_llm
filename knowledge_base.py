import os
import sentry_sdk

from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv


class KnowledgeBase:
    def __init__(self):
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
        self.BASE_URL = os.getenv("BASE_URL")
        self.LLM_MODEL = os.getenv("LLM_MODEL")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
        self.DOCS_FOLDER = os.getenv("DOCS_FOLDER")
        self.DEFAULT_DOCUMENT = os.getenv("DEFAULT_DOCUMENT")
        self.CONNECTION_STRING = os.getenv("CONNECTION_STRING")

        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME")

    def create_knowledge_base_pg(self, document_path):
        try:
            sentry_sdk.profiler.start_profiler()

            with sentry_sdk.start_transaction(op="task", name="Create knowledge"):
                with sentry_sdk.start_span(description="Store knowledge in pgvector"):
                    connection = self.CONNECTION_STRING
                    collection_name = self.COLLECTION_NAME

                    # loader = UnstructuredWordDocumentLoader(document_path)
                    if not os.path.exists(document_path):
                        return {
                            "error": f"Document path does not exist: {document_path}"
                        }

                    # Detect the file type and load the document
                    if document_path.endswith(".txt"):
                        with open(document_path, "r", encoding="utf-8") as file:
                            data = file.read()
                            # print(data)  # Check if the file loads correctly

                        loader = TextLoader(document_path, encoding="utf-8")
                        documents = loader.load()
                    elif document_path.endswith(".docx"):
                        loader = UnstructuredWordDocumentLoader(document_path)
                        documents = loader.load()


                    text_splitter = CharacterTextSplitter(
                        separator="\n", chunk_size=1000, chunk_overlap=100
                    )

                    text_chunks = text_splitter.split_documents(documents)

                    embeddings = OllamaEmbeddings(
                        base_url=self.BASE_URL, model=self.EMBEDDING_MODEL
                    )

                    vector_store = PGVector.from_documents(
                        embedding=embeddings,
                        connection=connection,
                        collection_name=collection_name,
                        documents=text_chunks,
                        use_jsonb=True,
                        create_extension=True,
                        embedding_length=768,
                        distance_strategy="cosine",
                        engine_args={"pool_size": 10, "max_overflow": 20},
                    )

                    print("vector store created")

            sentry_sdk.profiler.stop_profiler()

            return vector_store
        except Exception as e:
            print(f"Error creating knowledge base: {e}")
            return {"error": f"Error creating knowledge base: {e}"}

    def get_knowledge_base_pg(self):
        try:

            sentry_sdk.profiler.start_profiler()

            with sentry_sdk.start_transaction(op="task", name="Get knowledge"):
                with sentry_sdk.start_span(description="Get knowledge from pgvector"):
                    connection = self.CONNECTION_STRING
                    collection_name = self.COLLECTION_NAME

                    embeddings = OllamaEmbeddings(
                        base_url=self.BASE_URL, model=self.EMBEDDING_MODEL
                    )

                    knowledge_base = PGVector(
                        connection=connection,
                        collection_name=collection_name,
                        embeddings=embeddings,
                        use_jsonb=True,
                        create_extension=True,
                        embedding_length=768,
                        distance_strategy="cosine",
                        engine_args={"pool_size": 10, "max_overflow": 20},
                    )
                sentry_sdk.profiler.stop_profiler()

                return knowledge_base

        except Exception as e:
            print(f"Error while getting knowledge base: {e}")
            return {"error": f"Error while getting knowledge base: {e}"}


if __name__ == "__main__":
    kb = KnowledgeBase()
    # Create knowledge base for all documents in training folder

    for file in os.listdir(kb.DOCS_FOLDER):
        if file.endswith(".docx"):
            document_path = os.path.join(kb.DOCS_FOLDER, file)
            kb.create_knowledge_base_pg(document_path)

    # kb.get_knowledge_base_pg()
