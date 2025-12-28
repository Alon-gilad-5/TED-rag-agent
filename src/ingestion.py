import os
import pandas as pd
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()

# Configuration settings
INDEX_NAME = "ted-rag-index"
CSV_FILE = "../data/ted_talks_en.csv"
SUBSET_SIZE = 4005

# Chunking hyperparameters
CHUNK_SIZE = 1024
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.2)

# Batching settings for resume capability
BATCH_SIZE = 500  # Upload in batches of 500 chunks

# Pull API credentials from environment variables
azure_endpoint = os.getenv("BASE_URL", "https://api.llmod.ai")
azure_api_key = os.getenv("LLMOD_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


def get_existing_vector_count(pc, index_name):
    """Check how many vectors are already in Pinecone."""
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return stats.total_vector_count
    except Exception as e:
        print(f"Could not get index stats: {e}")
        return 0


def run_ingestion():
    """
    Resume-capable ingestion pipeline with batching and progress logging.
    """

    if not azure_api_key:
        raise ValueError("LLMOD_API_KEY not found in environment")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in environment")

    # Set up embedding model
    embeddings = AzureOpenAIEmbeddings(
        model="RPRTHPB-text-embedding-3-small",
        azure_deployment="RPRTHPB-text-embedding-3-small",
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        check_embedding_ctx_length=False
    )

    # Connect to Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created. Waiting for it to be ready...")
        time.sleep(10)
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    # Check how many vectors already exist
    existing_count = get_existing_vector_count(pc, INDEX_NAME)
    print(f"Vectors already in Pinecone: {existing_count}")

    # Load CSV
    print(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found in current directory.")
        return

    print(f"Dataset has {len(df)} talks total.")

    df_subset = df.head(SUBSET_SIZE)
    print(f"Processing subset of {len(df_subset)} talks...")

    # Create all document chunks
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for idx, row in df_subset.iterrows():
        meta = {
            "talk_id": str(row.get("talk_id", "")),
            "title": str(row.get("title", "")),
            "speaker_1": str(row.get("speaker_1", "")),
            "topics": str(row.get(("topics", ""))),
            "description": str(row.get("description", "")),
            "url": str(row.get("url", ""))
        }

        transcript = str(row.get("transcript", ""))
        title = str(row.get("title", ""))
        if transcript and transcript != "nan":
            enriched_text = f"Title: {title}. {transcript}"
            chunks = text_splitter.split_text(enriched_text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=meta))

    total_chunks = len(documents)
    print(f"Created {total_chunks} chunks from {len(df_subset)} talks.")

    # Skip already uploaded chunks
    if existing_count >= total_chunks:
        print("All chunks already uploaded. Nothing to do.")
        return

    documents_to_upload = documents[existing_count:]
    print(f"Resuming from chunk {existing_count}. Remaining: {len(documents_to_upload)} chunks.")

    # Upload in batches with progress logging
    if documents_to_upload:
        total_batches = (len(documents_to_upload) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(documents_to_upload), BATCH_SIZE):
            batch = documents_to_upload[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1

            try:
                print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                start_time = time.time()

                PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    index_name=INDEX_NAME,
                    pinecone_api_key=pinecone_api_key
                )

                elapsed = time.time() - start_time
                uploaded_so_far = existing_count + i + len(batch)
                print(
                    f"  ✓ Batch {batch_num} complete in {elapsed:.1f}s. Total uploaded: {uploaded_so_far}/{total_chunks}")

            except Exception as e:
                print(f"  ✗ Batch {batch_num} failed: {e}")
                print(f"  Resume point: {existing_count + i} chunks")
                print("  Re-run this script to continue from where it stopped.")
                return

        print("\n" + "=" * 50)
        print("Ingestion Complete!")
        print(f"Total vectors in Pinecone: {total_chunks}")
        print("=" * 50)
    else:
        print("No documents to upsert.")


if __name__ == "__main__":
    run_ingestion()