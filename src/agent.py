import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Keys
azure_endpoint = os.getenv("BASE_URL", "https://api.llmod.ai")
azure_api_key = os.getenv("LLMOD_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_version = os.getenv("OPENAI_API_VERSION")

if not azure_api_key:
    raise ValueError("Error: LLMOD_API_KEY is missing")
if not pinecone_api_key:
    raise ValueError("Error: PINECONE_API_KEY is missing")

# -- Config & Hyper-params --
RAG_CONFIG = {
    "chunk_size": 1024,
    "overlap_ratio": 0.2,
    "top_k": 5
}


class TedRagAgent:
    def __init__(self):
        # 1. Init embedding model
        self.embeddings = AzureOpenAIEmbeddings(
            model="RPRTHPB-text-embedding-3-small",
            azure_deployment="RPRTHPB-text-embedding-3-small",
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=openai_version,
            check_embedding_ctx_length=False
        )

        # 2. Init Pinecone vector store
        self.index_name = "ted-rag-index"
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_api_key
        )

        # 3. Init Azure LLM
        self.llm = AzureChatOpenAI(
            azure_deployment="RPRTHPB-gpt-5-mini",
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            temperature=1,
            max_tokens=1000,
        )

        # Strict system prompt (as required by assignment)
        self.system_prompt_text = (
            "You are a TED Talk assistant that answers questions strictly and only based on "
            "the TED dataset context provided to you (metadata and transcript passages). "
            "You must not use any external knowledge, the open internet, or information that "
            "is not explicitly contained in the retrieved context. "
            "If the answer cannot be determined from the provided context, respond: "
            "\"I don't know based on the provided TED data.\" "
            "Always explain your answer using the given context, quoting or paraphrasing "
            "the relevant transcript or metadata when helpful."
        )

    def get_response(self, user_query: str) -> dict:
        """Handle user query and return strict JSON format per assignment spec."""

        # 1. Retrieve relevant chunks with similarity scores
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            user_query,
            k=RAG_CONFIG['top_k']
        )

        context_text_list = []
        context_json_list = []

        for doc, score in docs_and_scores:
            title = doc.metadata.get("title", "Unknown")
            speaker = doc.metadata.get("speaker_1", "Unknown")  # Fixed: was "speaker1"

            # Format context for LLM
            content_snippet = (
                f"Title: {title}\n"
                f"Speaker: {speaker}\n"
                f"Transcript Snippet: {doc.page_content}\n"
            )
            context_text_list.append(content_snippet)

            # JSON for API response
            context_json_list.append({
                "talk_id": doc.metadata.get("talk_id", "N/A"),
                "title": title,
                "chunk": doc.page_content,
                "score": float(score)
            })

        full_context_block = "\n---\n".join(context_text_list)

        # 2. Build the user prompt
        user_prompt = f"Context:\n{full_context_block}\n\nQuestion: {user_query}"

        # 3. Create prompt template and invoke LLM
        messages = [
            ("system", self.system_prompt_text),
            ("user", user_prompt)
        ]
        prompt_template = ChatPromptTemplate.from_messages(messages)
        final_prompt_value = prompt_template.format_messages()

        ai_message = self.llm.invoke(final_prompt_value)
        response_text = ai_message.content

        # 4. Return response in required JSON format
        return {
            "response": response_text,
            "context": context_json_list,
            "Augmented_prompt": {
                "System": self.system_prompt_text,
                "User": user_prompt
            }
        }

    def get_status(self) -> dict:
        """Return RAG configuration as required by /api/stats endpoint."""
        return RAG_CONFIG


# Initialize agent singleton
rag_agent = TedRagAgent()
