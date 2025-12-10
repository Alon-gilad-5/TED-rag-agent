import pandas as pd
import json
import numpy as np
from typing import Dict, Any


def inspect_ted_data(csv_path: str, sample_size: int = 5):
    """
    Comprehensive TED dataset inspection for RAG planning
    """
    print("Loading TED dataset...")
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total talks: {len(df)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"Null values per column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts[null_counts > 0].items():
        print(f"  - {col}: {count} ({count / len(df) * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("TRANSCRIPT ANALYSIS (for chunking strategy)")
    print("=" * 60)

    # Calculate transcript statistics
    df['transcript_length'] = df['transcript'].fillna('').str.len()
    df['transcript_words'] = df['transcript'].fillna('').str.split().str.len()

    # Estimate tokens (rough: 1 token ≈ 0.75 words)
    df['estimated_tokens'] = (df['transcript_words'] * 1.33).astype(int)

    print(f"\nTranscript lengths:")
    print(f"  Characters: {df['transcript_length'].describe().to_dict()}")
    print(f"  Words: {df['transcript_words'].describe().to_dict()}")
    print(f"  Est. tokens: {df['estimated_tokens'].describe().to_dict()}")

    # Calculate chunking estimates
    chunk_sizes = [512, 1024, 1536, 2048]
    overlap_ratios = [0.1, 0.2, 0.3]

    print("\nChunking estimates (avg talk):")
    avg_tokens = df['estimated_tokens'].mean()
    for chunk_size in chunk_sizes:
        for overlap in overlap_ratios:
            effective_chunk = chunk_size * (1 - overlap)
            num_chunks = np.ceil(avg_tokens / effective_chunk)
            print(f"  Chunk={chunk_size}, Overlap={overlap}: ~{num_chunks:.0f} chunks/talk")

    print("\n" + "=" * 60)
    print("TOPICS FIELD ANALYSIS")
    print("=" * 60)

    # Analyze topics format
    sample_topics = df['topics'].head(sample_size)
    print(f"\nSample topics (first {sample_size}):")
    for i, topics in enumerate(sample_topics):
        print(f"  Talk {i + 1}: {topics[:100]}...")
        try:
            # Try parsing as JSON
            parsed = json.loads(topics)
            print(f"    -> Parsed as JSON: {type(parsed)}, {len(parsed)} items")
        except:
            print(f"    -> Not JSON, likely string format")

    print("\n" + "=" * 60)
    print("METADATA DISTRIBUTIONS")
    print("=" * 60)

    # Views distribution
    print(f"\nViews statistics:")
    print(f"  Mean: {df['views'].mean():,.0f}")
    print(f"  Median: {df['views'].median():,.0f}")
    print(f"  Max: {df['views'].max():,.0f}")
    print(f"  Top 10% threshold: {df['views'].quantile(0.9):,.0f}")

    # Duration distribution
    print(f"\nDuration (seconds):")
    print(f"  Mean: {df['duration'].mean() / 60:.1f} minutes")
    print(f"  Min: {df['duration'].min() / 60:.1f} minutes")
    print(f"  Max: {df['duration'].max() / 60:.1f} minutes")

    # Speaker analysis
    print(f"\nSpeakers:")
    print(f"  Unique primary speakers: {df['speaker_1'].nunique()}")
    print(f"  Talks with multiple speakers: {(df['all_speakers'] != df['speaker_1']).sum()}")

    print("\n" + "=" * 60)
    print("SAMPLE DATA FOR TESTING")
    print("=" * 60)

    # Select diverse sample for initial testing
    sample_talks = pd.DataFrame()

    # Get talks from different view ranges
    if len(df) > 100:
        high_views = df.nlargest(10, 'views').sample(3)
        medium_views = df[(df['views'] > df['views'].quantile(0.4)) &
                          (df['views'] < df['views'].quantile(0.6))].sample(3)
        low_views = df.nsmallest(50, 'views').sample(3)
        sample_talks = pd.concat([high_views, medium_views, low_views])
    else:
        sample_talks = df.sample(min(9, len(df)))

    print(f"\nRecommended test sample ({len(sample_talks)} talks):")
    for _, talk in sample_talks.iterrows():
        print(f"  - {talk['title'][:50]}... by {talk['speaker_1']}")
        print(f"    Views: {talk['views']:,}, Tokens: ~{talk['estimated_tokens']:,}")

    # Save sample for testing
    sample_talks.to_csv('ted_sample.csv', index=False)
    print(f"\n Sample saved to 'ted_sample.csv' for testing")

    print("\n" + "=" * 60)
    print("COST ESTIMATION")
    print("=" * 60)

    total_tokens = df['estimated_tokens'].sum()
    embedding_cost_per_1k = 0.00002  # text-embedding-3-small

    print(f"\nEmbedding cost estimates:")
    print(f"  Full dataset: ${total_tokens / 1000 * embedding_cost_per_1k:.2f}")
    print(f"  Sample (9 talks): ${sample_talks['estimated_tokens'].sum() / 1000 * embedding_cost_per_1k:.4f}")

    # print("\n RECOMMENDATIONS:")
    # print("  1. Start with sample (9 talks) for parameter tuning")
    # print("  2. Use chunk_size=1024 with overlap=0.2 as baseline")
    # print("  3. Parse topics field for better filtering")
    # print("  4. Consider caching embeddings locally during dev")


# Usage
if __name__ == "__main__":
    # Update path to your dataset
    inspect_ted_data('../data/ted_talks_en.csv')