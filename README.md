# LLM RAG Evaluation

Evaluation of a small language model with and without retrieval-augmented generation (RAG), using TriviaQA, ChromaDB retrieval, manual rubrics, retrieval metrics, failure analysis, and faithfulness evaluation.

## Project objective

This project asks a practical evaluation question: **does adding retrieval improve a small LLM's question-answering performance, and where does the RAG pipeline still fail?**

The same set of 20 TriviaQA validation questions was evaluated in two conditions:

1. **Standalone Q&A:** question → Qwen → answer
2. **RAG Q&A:** question → ChromaDB retrieval → retrieved context + question → the same Qwen model → answer

Keeping the model and questions fixed makes the comparison focused on the effect of retrieval.

## Stack

- Python
- Hugging Face `datasets` and `transformers`
- `Qwen/Qwen2.5-0.5B-Instruct`
- ChromaDB
- ChromaDB default embedding model (`all-MiniLM-L6-v2`)
- TriviaQA

The project was run locally on CPU.

## Dataset and corpus

The standalone baseline used the `rc.nocontext` configuration of TriviaQA. The RAG corpus used the corresponding `rc` examples so that search-result and Wikipedia evidence could be indexed.

For the 20 evaluation questions, evidence documents were collected from:

- `search_results.search_context`
- `entity_pages.wiki_context`

This produced **108 source documents**. Documents were split into 200-word chunks without overlap, producing **1,127 chunks** for the ChromaDB collection.

Retrieval searched the **entire 1,127-chunk collection**. It was deliberately not filtered by the question's provenance index, which would leak evaluation information into retrieval.

## Evaluation framework

### Answer-quality rubric

Standalone and RAG answers were manually evaluated on four independent 0–3 dimensions. No overall score was created.

| Dimension | What it measures |
|---|---|
| Factual correctness | Whether the answer is correct relative to the reference answer |
| Relevance | Whether the response addresses the question without substantial digression |
| Conciseness | Whether the response is appropriately brief and focused |
| Clarity | Whether the response is understandable and unambiguous |

For factual correctness, for example, `3` represents fully correct, `2` a correct core answer with a minor error, `1` partly correct with a major error, and `0` incorrect/no correct answer.

### Retrieval evaluation

Each question retrieved the top 3 chunks. The first rank containing genuine answer-bearing evidence was manually identified. Keyword or name overlap alone did not count as relevant evidence.

Two retrieval metrics were then calculated:

- **Answer Recall@3:** proportion of questions for which at least one of the top 3 chunks contained answer-bearing evidence.
- **MRR (Mean Reciprocal Rank):** rewards retrieval systems that place the first answer-bearing chunk higher in the ranking.

Because a complete set of every relevant document was not labeled, the project uses the term **Answer Recall@3** rather than claiming conventional document-level recall.

### Failure taxonomy

Failures were also labeled independently using a multi-label taxonomy:

- wrong factual answer
- hallucinated information
- irrelevant content
- excessive verbosity
- ambiguous/confusing response
- failure to answer

A response could have multiple failure labels, so category counts are not expected to sum to 20.

### Faithfulness

RAG answers were additionally scored for **faithfulness/groundedness** on a 0–3 scale. This measures whether factual claims in the generated answer are supported by the retrieved context. It is deliberately separate from factual correctness: an answer can contain a correct core answer while adding unsupported claims, or make a claim supported by a retrieved passage that nevertheless does not correctly answer the question.

## Results

### Baseline vs RAG answer quality

| Dimension (0–3) | Standalone baseline | RAG | Change |
|---|---:|---:|---:|
| Factual correctness | 0.35 | **1.25** | **+0.90** |
| Relevance | **1.95** | 1.80 | -0.15 |
| Conciseness | 1.10 | **1.45** | +0.35 |
| Clarity | 1.85 | **2.20** | +0.35 |

The largest improvement was factual correctness. RAG also improved conciseness and clarity in this sample, while relevance decreased slightly.

### Retrieval performance

| Metric | Result |
|---|---:|
| Answer Recall@3 | **0.85 (17/20)** |
| MRR | **0.733** |

The retriever therefore supplied answer-bearing evidence within the top three chunks for 85% of the questions.

### Retriever → generator handoff

A final answer was treated as substantially correct when its factual-correctness score was 2 or 3.

- Successful retrievals: **17**
- Substantially correct answers after successful retrieval: **7**
- Generator success given successful retrieval: **41.2% (7/17)**

This distinction was one of the most informative results of the project. Retrieval frequently succeeded, but the small generator often failed to convert available evidence into a sufficiently correct final answer.

### Failure analysis

| Failure category | Baseline | RAG |
|---|---:|---:|
| Wrong factual answer | 16/20 | **11/20** |
| Hallucinated information | 15/20 | **14/20** |
| Irrelevant content | **5/20** | 13/20 |
| Excessive verbosity | 15/20 | 15/20 |
| Ambiguous/confusing | 6/20 | 7/20 |
| Failure to answer | 1/20 | 1/20 |

RAG reduced the number of wrong factual answers, but hallucination remained frequent. Irrelevant content increased substantially, often because the small model generated explanatory or meta-text beyond the requested concise answer.

### Faithfulness

Average RAG faithfulness was **1.25/3**.

This reinforces the retriever/generator result: providing useful evidence does not guarantee that a model will remain grounded in that evidence.

## Main finding

The experiment demonstrates why RAG should be evaluated as a **pipeline**, rather than judged only from final-answer accuracy.

In this experiment, retrieval was relatively successful (**85% Answer Recall@3**), yet the model produced a substantially correct answer in only **41.2% of successful-retrieval cases**. RAG improved average factual correctness from **0.35 to 1.25**, but the model continued to hallucinate and frequently introduced irrelevant material.

The main bottleneck in this setup was therefore not simply finding relevant evidence. A major limitation was the small generator's ability to use retrieved evidence correctly and remain grounded.

## Example of why separate metrics matter

For the question *"Who was the man behind The Chipmunks?"*, retrieval placed answer-bearing evidence about **David Seville** in the top 3 and the model gave the correct core answer. However, it then added unsupported/incorrect claims. A single correctness metric would hide part of this behavior; separate retrieval, correctness, failure, and faithfulness measurements expose it.

## Limitations

This is a deliberately small evaluation study rather than a benchmark claim. Important limitations include:

- only 20 TriviaQA validation questions
- one small generator model (`Qwen2.5-0.5B-Instruct`)
- top-3 retrieval only
- fixed 200-word chunks with no overlap
- ChromaDB's default embedding setup
- manual relevance, answer-quality, failure, and faithfulness judgments
- no inter-rater agreement measurement
- generation behavior can vary with decoding/library configuration

The numerical results should therefore be interpreted as findings for this experiment, not as general performance estimates for Qwen or RAG systems.

## Reproducibility and evaluation design

The project uses fixed questions, the same generator for baseline and RAG conditions, separate scoring dimensions, explicit failure categories, and saved evaluation outputs. Raw model outputs are preserved rather than edited after generation. Retrieval provenance (`question_index`) is retained for evaluation but is not used as a retrieval filter.

A useful extension would be to rerun the fixed evaluation set after changing one component at a time—for example model size, chunking strategy, `n_results`, embeddings, or prompt—and compare each change against this baseline.

## Skills demonstrated

This project demonstrates practical work in:

- LLM evaluation design
- manual rubric-based evaluation
- Hugging Face model inference and datasets
- RAG pipeline construction
- document chunking and vector indexing
- ChromaDB semantic retrieval
- retrieval evaluation with Answer Recall@3 and MRR
- error taxonomy and failure analysis
- faithfulness/groundedness evaluation
- separating retrieval failures from generation failures
- reproducible experiment design

## Repository status

The evaluation was developed interactively in Jupyter/VS Code. The quantitative results reported above are the completed results of that evaluation. Supporting experiment outputs include baseline evaluation data/scores/failure analysis and RAG evaluation data, retrieval judgments, scores, failure analysis, pipeline metrics, and faithfulness scores.
