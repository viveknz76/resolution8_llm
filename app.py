
from flask import Flask, request, jsonify
from ollama_rag import RAGApp
import json

app = Flask(__name__)

@app.route("/ai", methods=["POST"])
def ollama_qa():
    data = request.json
    chat_history = data.get("chat_history")
    question = data.get("question")

    rag_app = RAGApp()

    response = rag_app.run_retrieval_chain(question, chat_history)

    return jsonify(response)


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
