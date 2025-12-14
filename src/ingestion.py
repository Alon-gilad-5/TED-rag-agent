import os
import pandas as pd
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Configuration settings
INDEX_NAME = "ted-rag-index"  # Name of our Pinecone vector database
CSV_FILE = "../data/ted_talks_en.csv"
SUBSET_SIZE = 200  # Only process this many talks (saves money on API costs)

# Chunking hyperparameters - how we split up long transcripts
CHUNK_SIZE = 1024  # Each chunk is about 1024 characters
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.2)  # 20% overlap between chunks so we don't lose context at boundaries

# Pull API credentials from environment variables
azure_endpoint = os.getenv("BASE_URL", "https://api.llmod.ai")
azure_api_key = os.getenv("LLMOD_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


def run_ingestion():
    """
    This is the main data pipeline that does everything:
    Load TED talks from CSV -> Split into chunks -> Turn into vectors -> Store in Pinecone

    Run this script once to populate your vector database with TED talk data!
    """

    # First, make sure we have all the API keys we need
    if not azure_api_key:
        raise ValueError("LLMOD_API_KEY not found in environment")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in environment")

    # Set up our embedding model - this turns text into vectors that capture meaning
    embeddings = AzureOpenAIEmbeddings(
        model="RPRTHPB-text-embedding-3-small",
        azure_deployment="RPRTHPB-text-embedding-3-small",
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        check_embedding_ctx_length=False  # Let the model handle text that's too long
    )

    # Connect to Pinecone and make sure our index exists
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if we already created the index before
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        # First time running? Let's create the index
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # text-embedding-3-small outputs 1536-dimensional vectors
            metric="cosine",  # Cosine similarity
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created. Waiting for it to be ready...")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    # Load up all the TED talks from our CSV file
    print(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found in current directory.")
        return

    print(f"Dataset has {len(df)} talks total.")

    # We're only processing a subset to keep costs down (embeddings cost money!)
    df_subset = df.head(SUBSET_SIZE)
    print(f"Processing subset of {len(df_subset)} talks...")

    # Time to chunk up the transcripts! Can't feed entire talks to the model at once.
    documents = []
    # This splitter is smart - it tries to split at natural boundaries (paragraphs, sentences, etc.)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # Max 1024 chars per chunk
        chunk_overlap=CHUNK_OVERLAP,  # 20% overlap so we don't lose context
        separators=["\n\n", "\n", ". ", " ", ""]  # Try these breaks in order
    )

    # Go through each TED talk and split it up
    for idx, row in df_subset.iterrows():
        # Grab the metadata we care about (talk title, speaker, etc.)
        meta = {
            "talk_id": str(row.get("talk_id", "")),
            "title": str(row.get("title", "")),
            "speaker_1": str(row.get("speaker_1", "")),
            "topics": str(row.get(("topics", ""))),
            "description": str(row.get("description", "")),
            "url": str(row.get("url", ""))
        }

        # Get the transcript and split it into bite-sized chunks
        transcript = str(row.get("transcript", ""))
        title = str(row.get("title", ""))
        if transcript and transcript != "nan":  # Make sure there's actually text to process
            enriched_text = f"Title: {title}. {transcript}"
            chunks = text_splitter.split_text(enriched_text)
            # Each chunk becomes its own document with the same metadata
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=meta))

    print(f"Created {len(documents)} chunks from {len(df_subset)} talks.")

    # Final step: upload everything to Pinecone
    if documents:
        print(f"Upserting {len(documents)} chunks to Pinecone...")
        # This will embed all the chunks and store them in the vector database
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=INDEX_NAME,
            pinecone_api_key=pinecone_api_key
        )
        print("Ingestion Complete!")
    else:
        print("No documents to upsert. Check if transcripts exist in CSV.")


# Run the ingestion when this script is executed directly
if __name__ == "__main__":
    run_ingestion()
