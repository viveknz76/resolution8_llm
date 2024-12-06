import os
from dotenv import load_dotenv
import requests
import sentry_sdk
import sentry_sdk.profiler
import concurrent.futures

from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from knowledge_base import KnowledgeBase


class RAGApp:
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

        self.knowledge_base = KnowledgeBase()

    # Retrieval grader to evaluate if the knowledge is sufficient to answer a question
    def retrieval_grader(self, question, llm, retriever_pg):
        prompt = PromptTemplate(
            template="""You are a grader assessing the relevance of a retrieved document to a user question.
        Your goal is to evaluate whether the document contains keywords or concepts related to the user question. It does not need to be a strict match.

        Instructions:
        - If the retrieved document is relevant to the question, output: {{"score": "yes"}}
        - If the retrieved document is NOT relevant to the question, output: {{"score": "no"}}

        Strictly adhere to the format: a JSON object with a single key "score" and a value of "yes" or "no". No additional text, explanations, or formatting is allowed.

        Here is the retrieved document: 
        {retrieved_context}

        Here is the user question: 
        {question}
        """,
            input_variables=["question", "retrieved_context"],
        )

        try:

            # Create a chain to grade the relevance of the retrieved context
            retrieval_grader = prompt | llm | JsonOutputParser()

            # Create a chain to retrieve the context
            docs = retriever_pg.invoke(question)

            generation = retrieval_grader.invoke(
                {"question": question, "retrieved_context": docs}
            )
            parsed_output = output_parser.parse(generation)
        except Exception as e:
            print(f"Error during retrieval grading: {e}")

    def run_retrieval_chain(
        self,
        question: str,
        chat_history: any,
        llm_model=None,
        base_url=None,
    ) -> any:
        try:
            llm_model = llm_model or self.LLM_MODEL
            base_url = base_url or self.BASE_URL

            sentry_sdk.profiler.start_profiler()

            with sentry_sdk.start_transaction(op="task", name="Run retrieval chain"):
                with sentry_sdk.start_span(description="Generate answer"):
                    # Set up the LLM
                    llm = ChatOllama(
                        base_url=base_url,
                        model=llm_model,
                        temperature=0,
                        callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
                        tags=["vivek_pc", "llama3.2", "ollama"],
                        disable_streaming=False,
                        cache=True,
                        verbose=True,
                        keep_alive=500,
                    )

                    # Create the knowledge base
                    # This should be replaced with a more robust way of creating the knowledge base in the future
                    default_document_path = os.path.join(
                        self.DOCS_FOLDER, self.DEFAULT_DOCUMENT
                    )

                    with sentry_sdk.start_span(description="Obtain knowledge base"):
                        knowledge_base_pg = self.knowledge_base.get_knowledge_base_pg()

                    if not knowledge_base_pg:
                        return {"answer": "Knowledge base missing."}

                    # Create retriever
                    with sentry_sdk.start_span(description="Create retriever"):
                        retriever_pg = knowledge_base_pg.as_retriever(
                            search_kwargs={"k": 3}
                        )

                    # Call the retrieval grader
                    # self.retrieval_grader(question, llm, retriever_pg)

                    with sentry_sdk.start_span(description="Create prompt"):
                        raw_prompt, retriever_prompt = self.create_prompt()

                    # Create question-answer chain
                    with sentry_sdk.start_span(
                        description="Create question-answer chain"
                    ):
                        question_answer_chain = create_stuff_documents_chain(
                            llm, raw_prompt
                        )
                        print(question_answer_chain)

                    # Create history-aware retriever
                    with sentry_sdk.start_span(
                        description="Create history-aware retriever"
                    ):
                        history_aware_retriever = create_history_aware_retriever(
                            llm=llm, retriever=retriever_pg, prompt=retriever_prompt
                        )

                    # Create and run RAG chain
                    with sentry_sdk.start_span(description="Create and run RAG chain"):
                        rag_chain = create_retrieval_chain(
                            history_aware_retriever, question_answer_chain
                        )

                    # Get the answer
                    with sentry_sdk.start_span(description="Get answer"):
                        chain = rag_chain.pick("answer")
                        for chunk in chain.stream({"input": question}):
                            yield chunk

                    yield chunk

            sentry_sdk.profiler.stop_profiler()

        except Exception as e:
            print(e)
            return {"answer": f"An error occurred while generating the answer. {e}"}

    def create_prompt(self):
        # system_prompt = (
        #     "You are an assistant for question-answering tasks. "
        #     "You specialize in answering questions related to object storage solutions. "
        #     "Use the following pieces of retrieved context to answer "
        #     "the question. You can compare the knowledge base to other object storage providers to answer questions. "
        #     "Use pricing models of the competitors for comparison when required. "
        #     "If you don't know the answer, say that you don't know."
        #     "\n\n"
        #     "{context}"
        # )

        system_prompt = (
            "You specialize in answer questions about Resolution8 limited based in Hamilton, New Zealand. Only use the context provided to answer the questions. "
            "When some asks 'Tell me about Resolution8', provide information on Resolution8 Limited from the context. "
            "Do not use any references to Resolution8 Limited in UK or any other entity. "
            "Resolution8 is defined as a company specializing in project management, business intelligence, software development, and data analytics in Hamilton, New Zealand. "
            "Encourage the user to ask more specific questions about Resolution8. "
            "And keep the tone neutral and professional. "
            "Avoid references to personal development, James Clear, or any unrelated entity. "
            "If you don't know the answer, say that you don't know. "
            "\n\n"
            "{context}"
        )

        raw_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        retriever_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        return raw_prompt, retriever_prompt

    # Write a function to prime the llama model hosted on the server
    def prime_llama(self):
        try:
            base_url = self.BASE_URL
            payload = {"model": self.LLM_MODEL, "text": "Hello!"}
            url = f"{base_url}/api/chat"
            response = requests.post(url, json=payload)
            response.raise_for_status()

            print(f"LLM primed successfully. {response.status_code}")

            return {"message": "LLM primed successfully."}
        except Exception as e:
            print(f"Error priming LLM: {e}")
            return {"error": f"Error priming LLM: {e}"}

    def prime_llama_in_background(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.prime_llama)
