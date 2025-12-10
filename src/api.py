from flask import Flask, request, jsonify
from agent import rag_agent  # Imports the agent we created

app = Flask(__name__)


@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    """
    Endpoint 1: POST /api/prompt [cite: 63]
    Input: {"question": "..."}
    Output: JSON with response, context, and Augmented_prompt
    """
    data = request.get_json()

    # Validation
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    user_query = data['question']

    try:
        # Get the answer from our Agent
        result = rag_agent.get_response(user_query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def handle_stats():
    """
    Endpoint 2: GET /api/stats [cite: 89]
    Returns the RAG configuration (chunk_size, etc.)
    """
    return jsonify(rag_agent.get_status())


@app.route('/', methods=['GET'])
def health_check():
    """Simple health check to see if server is running"""
    return "TED Talk RAG Agent is running!", 200


if __name__ == '__main__':
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=True)