from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>TED Talk RAG Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        header {
            text-align: center;
            margin-bottom: 40px;
        }
        h1 { 
            color: #fff;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        h1 span { color: #e62b1e; }
        .subtitle { color: #8892b0; font-size: 1.1rem; }
        .search-box {
            background: #fff;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
        }
        textarea {
            width: 100%;
            height: 100px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px;
            font-size: 1rem;
            font-family: inherit;
            resize: none;
            transition: border-color 0.3s;
        }
        textarea:focus { outline: none; border-color: #e62b1e; }
        button {
            background: linear-gradient(135deg, #e62b1e 0%, #c4251a 100%);
            color: white;
            padding: 14px 32px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(230,43,30,0.3); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        #loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #64748b;
        }
        .spinner {
            width: 40px; height: 40px;
            border: 4px solid #e2e8f0;
            border-top-color: #e62b1e;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        #response {
            display: none;
            margin-top: 25px;
            padding: 25px;
            background: #f8fafc;
            border-radius: 12px;
            border-left: 4px solid #e62b1e;
            line-height: 1.7;
            color: #334155;
            white-space: pre-wrap;
        }
        .endpoints {
            margin-top: 40px;
            text-align: center;
            color: #8892b0;
            font-size: 0.9rem;
        }
        .endpoints code {
            background: rgba(255,255,255,0.1);
            padding: 4px 8px;
            border-radius: 6px;
            margin: 0 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span>TED</span> Talk RAG Assistant</h1>
            <p class="subtitle">Ask questions about TED talks using AI-powered retrieval</p>
        </header>

        <div class="search-box">
            <textarea id="question" placeholder="e.g., Find a TED talk about education, creativity, or technology..."></textarea>
            <button id="askBtn" onclick="askQuestion()">Ask Question</button>

            <div id="loading">
                <div class="spinner"></div>
                <p>Searching TED talks...</p>
            </div>

            <div id="response"></div>
        </div>

        <div class="endpoints">
            API Endpoints: <code>POST /api/prompt</code> <code>GET /api/stats</code>
        </div>
    </div>
<script>
        async function askQuestion() {
            const question = document.getElementById('question').value;
            if (!question) return;

            const btn = document.getElementById('askBtn');
            btn.disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('response').style.display = 'none';

            try {
                const res = await fetch('/api/prompt', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question})
                });
                const text = await res.text();
                let output;
                try {
                    const data = JSON.parse(text);
                    output = data.response || data.error || text;
                } catch {
                    output = text;
                }
                document.getElementById('response').innerText = output;
                document.getElementById('response').style.display = 'block';
            } catch (e) {
                document.getElementById('response').innerText = 'Error: ' + e.message;
                document.getElementById('response').style.display = 'block';
            }
            document.getElementById('loading').style.display = 'none';
            btn.disabled = false;
        }

        document.getElementById('question').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) askQuestion();
        });
    </script>
</body>
</html>
        """
        self.wfile.write(html.encode())