# Basic imports for handling HTTP requests and working with JSON
from http.server import BaseHTTPRequestHandler
import json
import os
# LangChain stuff - helps us talk to Azure's OpenAI and search through our vector database
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Grab our API credentials from environment variables (keeps secrets out of the code)
AZURE_ENDPOINT = os.environ.get("BASE_URL", "https://api.llmod.ai")
AZURE_API_KEY = os.environ.get("LLMOD_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Configuration for our RAG system - basically how we chunk up and search through TED talks
RAG_CONFIG = {"chunk_size": 1024, "overlap_ratio": 0.2, "top_k": 5}

# This is the instruction we give to the AI - telling it to ONLY use the TED data we give it
# and not make stuff up from its training. Keeps answers grounded in actual TED content.
SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and only based on "
    "the TED dataset context provided to you (metadata and transcript passages). "
    "You must not use any external knowledge, the open internet, or information that "
    "is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    "\"I don't know based on the provided TED data.\" "
    "Always explain your answer using the given context, quoting or paraphrasing "
    "the relevant transcript or metadata when helpful."
)


# Main request handler - this is what processes incoming questions about TED talks
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Read the incoming request body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        # Try to parse the JSON and grab the user's question
        try:
            data = json.loads(body)
            question = data.get("question")
            if not question:
                self._send_error(400, "Missing 'question' field")
                return
        except json.JSONDecodeError:
            # If they sent us garbage, let them know
            self._send_error(400, "Invalid JSON")
            return

        # Set up the embedding model - this converts text into vectors for similarity search
        embeddings = AzureOpenAIEmbeddings(
            model="RPRTHPB-text-embedding-3-small",
            azure_deployment="RPRTHPB-text-embedding-3-small",
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            check_embedding_ctx_length=False
        )

        # Connect to our Pinecone vector database where all the TED talks are stored
        vectorstore = PineconeVectorStore(
            index_name="ted-rag-index",
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )

        # Initialize the language model that'll actually answer the question
        llm = AzureChatOpenAI(
            azure_deployment="RPRTHPB-gpt-5-mini",
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version="2024-02-15-preview",
            temperature=1,
            max_tokens=1000,
        )

        # Search for the most relevant TED talk chunks based on the question
        docs_and_scores = vectorstore.similarity_search_with_score(question, k=RAG_CONFIG['top_k'])

        # We'll build up context in two formats - one for the AI, one for the response
        context_text_list = []
        context_json_list = []

        # Go through each matching document and extract the good bits
        for doc, score in docs_and_scores:
            title = doc.metadata.get("title", "Unknown")
            speaker = doc.metadata.get("speaker_1", "Unknown")

            # Format for the AI to read
            context_text_list.append(f"Title: {title}\nSpeaker: {speaker}\nTranscript Snippet: {doc.page_content}\n")
            # Format for returning to the user (so they can see what we found)
            context_json_list.append({
                "talk_id": doc.metadata.get("talk_id", "N/A"),
                "title": title,
                "chunk": doc.page_content,
                "score": float(score)
            })

        # Combine all the context into one block of text
        full_context_block = "\n---\n".join(context_text_list)
        user_prompt = f"Context:\n{full_context_block}\n\nQuestion: {question}"

        # Build the full prompt with system instructions + context + question
        messages = [("system", SYSTEM_PROMPT), ("user", user_prompt)]
        prompt_template = ChatPromptTemplate.from_messages(messages)
        # Send it to the AI and get an answer back
        ai_message = llm.invoke(prompt_template.format_messages())

        # Package everything up nicely for the response
        result = {
            "response": ai_message.content,  # The AI's actual answer
            "context": context_json_list,  # The TED talks we used to answer
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt}  # Full prompt for debugging
        }

        # Send the successful response back
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    # Helper to send error responses when things go wrong
    def _send_error(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())