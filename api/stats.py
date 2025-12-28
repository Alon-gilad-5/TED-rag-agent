from http.server import BaseHTTPRequestHandler
import json

RAG_CONFIG = {
    "chunk_size": 1024,
    "overlap_ratio": 0.2,
    "top_k": 3 #choosen by evaluationRag.py
}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(RAG_CONFIG).encode())
