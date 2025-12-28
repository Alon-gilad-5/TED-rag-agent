"""
LLM-as-Judge Evaluation Script for TED Talk RAG System
Evaluates retrieval quality across different k values using the 4 query categories.
"""

import os
import json
from dataclasses import dataclass
from typing import List, Dict
from dotenv import load_dotenv

# Fix for langchain version mismatch
import langchain
langchain.verbose = False

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AZURE_ENDPOINT = os.environ.get("BASE_URL", "https://api.llmod.ai")
AZURE_API_KEY = os.environ.get("LLMOD_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# k values to test
K_VALUES = [3, 5, 10,12,15]

# RAG system prompt (same as your production system)
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

# =============================================================================
# TEST QUESTIONS - Based on the 4 Assignment Categories
# =============================================================================

TEST_QUESTIONS = [
    # Category 1: Precise Fact Retrieval
    {
        "category": "Precise Fact Retrieval",
        "question": "Find a TED talk that discusses overcoming fear or anxiety. Provide the title and speaker.",
        "evaluation_criteria": "Must return a specific talk title and speaker name that relates to fear/anxiety"
    },
    {
        "category": "Precise Fact Retrieval",
        "question": "Find a TED talk about the human brain. Provide the title and speaker.",
        "evaluation_criteria": "Must return a specific talk title and speaker name about neuroscience/brain"
    },

    # Category 2: Multi-Result Topic Listing
    {
        "category": "Multi-Result Topic Listing",
        "question": "Which TED talks focus on education or learning? Return a list of exactly 3 talk titles.",
        "evaluation_criteria": "Must return exactly 3 distinct talk titles related to education/learning"
    },
    {
        "category": "Multi-Result Topic Listing",
        "question": "List 3 TED talks about technology or innovation.",
        "evaluation_criteria": "Must return exactly 3 distinct talk titles related to technology/innovation"
    },

    # Category 3: Key Idea Summary Extraction
    {
        "category": "Key Idea Summary",
        "question": "Find a TED talk where the speaker talks about technology improving people's lives. Provide the title and a short summary of the key idea.",
        "evaluation_criteria": "Must provide talk title AND a coherent summary grounded in transcript evidence"
    },
    {
        "category": "Key Idea Summary",
        "question": "Find a TED talk about creativity. Provide the title and summarize the speaker's main argument.",
        "evaluation_criteria": "Must provide talk title AND a summary of the main argument from transcript"
    },

    # Category 4: Recommendation with Justification
    {
        "category": "Recommendation with Justification",
        "question": "I'm looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?",
        "evaluation_criteria": "Must recommend ONE talk with evidence-based justification from retrieved data"
    },
    {
        "category": "Recommendation with Justification",
        "question": "Recommend a TED talk for someone interested in leadership. Explain why this talk is relevant.",
        "evaluation_criteria": "Must recommend ONE talk with justification grounded in transcript/metadata"
    },
]

# =============================================================================
# LLM-AS-JUDGE PROMPT
# =============================================================================

JUDGE_PROMPT = """You are an impartial evaluator assessing a RAG system's response quality.

EVALUATION CRITERIA:
1. **Faithfulness (1-5)**: Is the answer grounded ONLY in the provided context? 
   - 5 = Fully grounded, quotes/paraphrases context appropriately
   - 1 = Makes claims not supported by context or hallucinates

2. **Relevance (1-5)**: Does the answer address the question asked?
   - 5 = Directly and completely answers the question
   - 1 = Irrelevant or misses the point

3. **Completeness (1-5)**: Does it fulfill all requirements (e.g., "list 3 talks", "provide title AND summary")?
   - 5 = All requirements met
   - 1 = Missing major requirements

4. **Context Quality (1-5)**: Were the retrieved chunks useful for answering?
   - 5 = Highly relevant chunks that enable a good answer
   - 1 = Irrelevant chunks that don't help

---
QUESTION: {question}

CATEGORY: {category}

SPECIFIC CRITERIA: {evaluation_criteria}

RETRIEVED CONTEXT:
{context}

RAG SYSTEM RESPONSE:
{response}

---
Provide your evaluation as JSON:
{{
    "faithfulness": <1-5>,
    "relevance": <1-5>,
    "completeness": <1-5>,
    "context_quality": <1-5>,
    "overall_score": <1-5>,
    "reasoning": "<brief explanation of scores>"
}}

Return ONLY valid JSON, no other text.
"""

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

@dataclass
class EvaluationResult:
    question: str
    category: str
    k_value: int
    response: str
    context_chunks: List[Dict]
    scores: Dict
    avg_similarity: float


def setup_components():
    """Initialize LLM, embeddings, and vector store."""
    embeddings = AzureOpenAIEmbeddings(
        model="RPRTHPB-text-embedding-3-small",
        azure_deployment="RPRTHPB-text-embedding-3-small",
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        check_embedding_ctx_length=False
    )

    vectorstore = PineconeVectorStore(
        index_name="ted-rag-index",
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )

    # FIXED: temperature=1 (only supported value for gpt-5-mini)
    llm = AzureChatOpenAI(
        azure_deployment="RPRTHPB-gpt-5-mini",
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version="2024-02-15-preview",
        temperature=1,
        max_tokens=1000,
    )

    # Judge LLM (same settings)
    judge_llm = AzureChatOpenAI(
        azure_deployment="RPRTHPB-gpt-5-mini",
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version="2024-02-15-preview",
        temperature=1,
        max_tokens=500,
    )

    return embeddings, vectorstore, llm, judge_llm


def generate_rag_response(vectorstore, llm, question: str, k: int) -> tuple:
    """Generate RAG response for a given question and k value."""
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=k)

    context_text_list = []
    context_json_list = []

    for doc, score in docs_and_scores:
        title = doc.metadata.get("title", "Unknown")
        speaker = doc.metadata.get("speaker_1", "Unknown")
        topics = doc.metadata.get("topics", "")

        context_text_list.append(
            f"Title: {title}\n"
            f"Speaker: {speaker}\n"
            f"Topics: {topics}\n"
            f"Transcript: {doc.page_content}\n"
        )

        context_json_list.append({
            "talk_id": doc.metadata.get("talk_id", "N/A"),
            "title": title,
            "speaker": speaker,
            "chunk": doc.page_content[:200] + "...",
            "score": float(score)
        })

    full_context = "\n---\n".join(context_text_list)
    user_prompt = f"Context:\n{full_context}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = llm.invoke(messages)
    avg_similarity = sum(d[1] for d in docs_and_scores) / len(docs_and_scores)

    return response.content, context_json_list, full_context, avg_similarity


def judge_response(judge_llm, question: str, category: str, criteria: str,
                   context: str, response: str) -> Dict:
    """Use LLM-as-Judge to evaluate the response."""
    judge_input = JUDGE_PROMPT.format(
        question=question,
        category=category,
        evaluation_criteria=criteria,
        context=context[:3000],
        response=response
    )

    try:
        judgment = judge_llm.invoke(judge_input)
        # Try to extract JSON from response
        content = judgment.content.strip()
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        scores = json.loads(content)
        return scores
    except (json.JSONDecodeError, Exception) as e:
        print(f"    Warning: Could not parse judge response: {e}")
        return {
            "faithfulness": 3,
            "relevance": 3,
            "completeness": 3,
            "context_quality": 3,
            "overall_score": 3,
            "reasoning": "Failed to parse judge response"
        }


def run_evaluation(k_values: List[int] = K_VALUES,
                   questions: List[Dict] = TEST_QUESTIONS) -> List[EvaluationResult]:
    """Run full evaluation across all k values and questions."""
    print("Setting up components...")
    embeddings, vectorstore, llm, judge_llm = setup_components()

    results = []
    total_tests = len(k_values) * len(questions)
    current = 0

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Testing k={k}")
        print('='*60)

        for q in questions:
            current += 1
            print(f"\n[{current}/{total_tests}] {q['category']}")
            print(f"Question: {q['question'][:60]}...")

            try:
                # Generate RAG response
                response, context_chunks, full_context, avg_sim = generate_rag_response(
                    vectorstore, llm, q['question'], k
                )

                # Judge the response
                scores = judge_response(
                    judge_llm,
                    q['question'],
                    q['category'],
                    q['evaluation_criteria'],
                    full_context,
                    response
                )

                result = EvaluationResult(
                    question=q['question'],
                    category=q['category'],
                    k_value=k,
                    response=response,
                    context_chunks=context_chunks,
                    scores=scores,
                    avg_similarity=avg_sim
                )
                results.append(result)

                print(f"  Overall Score: {scores.get('overall_score', 'N/A')}/5")
                print(f"  Avg Similarity: {avg_sim:.4f}")

            except Exception as e:
                print(f"  Error: {e}")
                # Add placeholder result
                results.append(EvaluationResult(
                    question=q['question'],
                    category=q['category'],
                    k_value=k,
                    response="ERROR",
                    context_chunks=[],
                    scores={"overall_score": 0},
                    avg_similarity=0
                ))

    return results


def analyze_results(results: List[EvaluationResult]) -> Dict:
    """Analyze results and find optimal k value."""
    analysis = {}

    for k in K_VALUES:
        k_results = [r for r in results if r.k_value == k and r.scores.get("overall_score", 0) > 0]

        if not k_results:
            continue

        avg_scores = {
            "faithfulness": sum(r.scores.get("faithfulness", 0) for r in k_results) / len(k_results),
            "relevance": sum(r.scores.get("relevance", 0) for r in k_results) / len(k_results),
            "completeness": sum(r.scores.get("completeness", 0) for r in k_results) / len(k_results),
            "context_quality": sum(r.scores.get("context_quality", 0) for r in k_results) / len(k_results),
            "overall": sum(r.scores.get("overall_score", 0) for r in k_results) / len(k_results),
            "avg_similarity": sum(r.avg_similarity for r in k_results) / len(k_results),
        }

        analysis[k] = avg_scores

    if analysis:
        optimal_k = max(analysis.keys(), key=lambda k: analysis[k]["overall"])
        return {
            "by_k_value": analysis,
            "optimal_k": optimal_k,
            "optimal_scores": analysis[optimal_k]
        }
    return {"by_k_value": {}, "optimal_k": 5, "optimal_scores": {}}


def print_summary(analysis: Dict):
    """Print formatted summary of results."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    print("\nScores by k value:")
    print("-"*70)
    print(f"{'k':>4} | {'Faith':>6} | {'Relev':>6} | {'Compl':>6} | {'Contxt':>6} | {'Overall':>7} | {'AvgSim':>7}")
    print("-"*70)

    for k, scores in analysis["by_k_value"].items():
        print(f"{k:>4} | {scores['faithfulness']:>6.2f} | {scores['relevance']:>6.2f} | "
              f"{scores['completeness']:>6.2f} | {scores['context_quality']:>6.2f} | "
              f"{scores['overall']:>7.2f} | {scores['avg_similarity']:>7.4f}")

    print("-"*70)
    print(f"\n✓ OPTIMAL k = {analysis['optimal_k']} (Overall Score: {analysis['optimal_scores'].get('overall', 0):.2f}/5)")
    print("\nRecommendation for your RAG_CONFIG:")
    print(f'  RAG_CONFIG = {{"chunk_size": 1024, "overlap_ratio": 0.2, "top_k": {analysis["optimal_k"]}}}')


def save_detailed_results(results: List[EvaluationResult], filepath: str = "evaluation_results.json"):
    """Save detailed results to JSON file."""
    output = []
    for r in results:
        output.append({
            "question": r.question,
            "category": r.category,
            "k_value": r.k_value,
            "response": r.response,
            "context_chunks": r.context_chunks,
            "scores": r.scores,
            "avg_similarity": r.avg_similarity
        })

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("TED Talk RAG Evaluation - LLM-as-Judge")
    print("="*50)

    results = run_evaluation()
    analysis = analyze_results(results)
    print_summary(analysis)
    save_detailed_results(results)