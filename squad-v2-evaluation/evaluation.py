"""SQuAD v2 baseline-vs-RAG evaluation.

This script captures the core pipeline used in the portfolio experiment.
Human rubric/failure labels are included as the completed evaluation data.
"""

from datasets import load_dataset
from transformers import pipeline
import chromadb

SELECTED_INDICES = [
    0, 208, 626, 950, 1160, 1407, 1831, 2275, 2690, 2945,
    3366, 3650, 3975, 4250, 4623, 4842, 5204, 5448, 5775, 6007,
]

CHUNK_SIZE = 300
OVERLAP = 50
N_RESULTS = 3

dataset = load_dataset("rajpurkar/squad_v2", split="validation")
evaluation_dataset = dataset.select(SELECTED_INDICES)

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
)

# Baseline generation
model_answers = []
for i in range(20):
    question = evaluation_dataset[i]["question"]
    output = generator(question, max_new_tokens=50, return_full_text=False)
    model_answers.append(output[0]["generated_text"])

reference_answers = []
for i in range(20):
    reference_answers.append(evaluation_dataset[i]["answers"]["text"][0])

# RAG corpus
contexts = []
for i in range(20):
    contexts.append(evaluation_dataset[i]["context"])

all_chunks = []
chunk_context_indices = []

for i in range(20):
    text = contexts[i]
    for j in range(0, len(text), CHUNK_SIZE - OVERLAP):
        chunk = text[j:j + CHUNK_SIZE]
        all_chunks.append(chunk)
        chunk_context_indices.append(i)

client = chromadb.Client()
collection = client.create_collection(name="squad_chunks")

chunk_ids = []
for i in range(len(all_chunks)):
    chunk_ids.append(str(i))

collection.add(documents=all_chunks, ids=chunk_ids)

# RAG generation
rag_answers = []
retrieved_ids_all = []

for i in range(20):
    current_question = evaluation_dataset[i]["question"]

    results = collection.query(
        query_texts=[current_question],
        n_results=N_RESULTS,
    )

    retrieved_ids_all.append(results["ids"][0])
    retrieved_chunks = results["documents"][0]
    rag_context = "\n\n".join(retrieved_chunks)

    rag_prompt = f"""Use the context below to answer the question.

Context:
{rag_context}

Question:
{current_question}

Answer:"""

    output = generator(
        rag_prompt,
        max_new_tokens=50,
        return_full_text=False,
    )
    rag_answers.append(output[0]["generated_text"])

# Retrieval evaluation
recall_results = []
reciprocal_ranks = []

for i in range(20):
    retrieved_ids = retrieved_ids_all[i]
    retrieved_contexts = []

    for chunk_id in retrieved_ids:
        retrieved_contexts.append(chunk_context_indices[int(chunk_id)])

    recall_results.append(1 if i in retrieved_contexts else 0)

    reciprocal_rank = 0
    for j in range(len(retrieved_ids)):
        chunk_id = int(retrieved_ids[j])
        if chunk_context_indices[chunk_id] == i:
            reciprocal_rank = 1 / (j + 1)
            break
    reciprocal_ranks.append(reciprocal_rank)

# Completed human evaluation labels
factual_scores = [2,0,0,0,0,0,3,1,0,1,2,0,0,0,1,3,3,1,2,2]
relevance_scores = [3,2,0,2,0,2,3,2,0,1,3,3,1,1,3,3,3,3,3,2]
conciseness_scores = [1,1,0,1,2,3,1,0,0,1,1,2,0,0,1,1,2,3,1,0]
clarity_scores = [2,2,1,2,3,2,3,2,1,2,3,2,2,1,2,3,3,3,3,1]

rag_factual_scores = [3,3,3,2,3,3,3,3,3,3,0,2,3,1,3,2,3,3,3,1]
rag_relevance_scores = [3,3,3,3,3,3,2,3,3,3,1,3,3,2,3,3,3,3,3,2]
rag_conciseness_scores = [2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,1,2,2]
rag_clarity_scores = [3,3,3,3,3,3,2,3,3,3,2,3,3,2,3,3,3,2,3,3]
faithfulness_scores = [2,3,3,2,3,3,1,3,3,3,1,2,3,0,3,1,2,1,3,2]

wrong_factual = [1,1,1,1,0,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0]
hallucination = [1,1,1,1,0,1,0,0,1,1,0,1,0,1,1,0,0,0,0,1]
irrelevant = [1,1,1,1,0,0,1,0,1,1,1,0,0,1,1,0,0,0,0,1]
verbosity = [1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1]
ambiguity_confusion = [1,0,1,1,0,1,0,0,1,1,0,1,0,1,1,0,0,0,0,1]
failure_to_answer = [0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0]

rag_wrong_factual = [0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1]
rag_hallucination = [0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,0]
rag_irrelevant = [0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1]
rag_verbosity = [1] * 20
rag_ambiguity_confusion = [0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,1]
rag_failure_to_answer = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]

project_results = {
    "baseline_factual": sum(factual_scores) / 20,
    "rag_factual": sum(rag_factual_scores) / 20,
    "baseline_relevance": sum(relevance_scores) / 20,
    "rag_relevance": sum(rag_relevance_scores) / 20,
    "baseline_conciseness": sum(conciseness_scores) / 20,
    "rag_conciseness": sum(rag_conciseness_scores) / 20,
    "baseline_clarity": sum(clarity_scores) / 20,
    "rag_clarity": sum(rag_clarity_scores) / 20,
    "recall_at_3": sum(recall_results) / 20,
    "mrr": sum(reciprocal_ranks) / 20,
    "faithfulness": sum(faithfulness_scores) / 20,
    "baseline_wrong_factual": sum(wrong_factual),
    "rag_wrong_factual": sum(rag_wrong_factual),
    "baseline_hallucination": sum(hallucination),
    "rag_hallucination": sum(rag_hallucination),
    "baseline_irrelevant": sum(irrelevant),
    "rag_irrelevant": sum(rag_irrelevant),
    "baseline_verbosity": sum(verbosity),
    "rag_verbosity": sum(rag_verbosity),
    "baseline_ambiguity": sum(ambiguity_confusion),
    "rag_ambiguity": sum(rag_ambiguity_confusion),
    "baseline_failure_to_answer": sum(failure_to_answer),
    "rag_failure_to_answer": sum(rag_failure_to_answer),
}

print(project_results)
