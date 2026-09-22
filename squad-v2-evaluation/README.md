# SQuAD v2 Baseline vs RAG Evaluation

A compact LLM evaluation study comparing standalone question answering with retrieval-augmented generation (RAG) on a fixed 20-question subset of SQuAD v2.

## Objective

The project asks two separate questions:

1. Does retrieval improve answer quality for a small language model?
2. When retrieval succeeds but the final answer fails, is the bottleneck retrieval or generation/context use?

The same model and the same 20 questions are used in both conditions.

## Stack

- Python
- Hugging Face `datasets`
- Hugging Face `transformers`
- `Qwen/Qwen2.5-0.5B-Instruct`
- ChromaDB
- SQuAD v2 validation split
- Local CPU inference

## Evaluation design

A fixed set of 20 SQuAD v2 validation examples with distinct source titles was selected. Baseline answers were generated from the question alone. For RAG, the 20 source contexts were split into overlapping character chunks (300 characters with 50-character overlap), indexed in ChromaDB, and the top 3 retrieved chunks were inserted into the prompt.

Answer quality was manually scored on four independent 0–3 dimensions:

- factual correctness
- relevance
- conciseness
- clarity

RAG answers were additionally scored for faithfulness to retrieved evidence. A multi-label failure taxonomy tracked wrong factual answers, hallucination, irrelevant content, verbosity, ambiguity/confusion, and failure to answer.

Retrieval was evaluated with Recall@3 and Mean Reciprocal Rank (MRR).

## Results

| Metric | Baseline | RAG |
|---|---:|---:|
| Factual correctness (0–3) | 1.05 | **2.50** |
| Relevance (0–3) | 2.00 | **2.75** |
| Conciseness (0–3) | 1.05 | **1.90** |
| Clarity (0–3) | 2.15 | **2.80** |

### Retrieval and faithfulness

| Metric | Result |
|---|---:|
| Recall@3 | **1.00** |
| MRR | **1.00** |
| Mean faithfulness (0–3) | **2.20** |

The retrieval scores should be interpreted cautiously: this is a small controlled corpus containing only contexts associated with the 20 evaluation questions. They demonstrate that retrieval worked correctly in this experiment, not that the retriever is universally strong.

### Failure analysis

| Failure category | Baseline | RAG |
|---|---:|---:|
| Wrong factual | 10/20 | **5/20** |
| Hallucination | 11/20 | **5/20** |
| Irrelevant content | 11/20 | **5/20** |
| Verbosity | 16/20 | 20/20 |
| Ambiguity/confusion | 10/20 | **7/20** |
| Failure to answer | 4/20 | **1/20** |

## Main finding

RAG substantially improved factual correctness and reduced most substantive failure categories. However, perfect source-context retrieval on this controlled corpus did not produce perfect answers. Faithfulness remained 2.20/3 and several answers failed despite the correct source passage being ranked first.

This separates **retrieval quality** from **generator/context-utilization quality**. The remaining bottleneck in this setup was primarily the generator's use of retrieved evidence and its tendency to produce unnecessary explanatory text.

Examples included:

- returning “Portuguese” when the retrieved Amazon passage explicitly gave “Amazonia or the Amazon Jungle”;
- claiming the retrieved context did not say what Paul Baran developed even though it explicitly did;
- answering “a volcanic eruption” when the retrieved geology passage stated “melt (magma and/or lava).”

## Actionable recommendations

1. Strengthen the prompt to require a short answer supported only by retrieved context.
2. Add an abstention rule when the evidence is insufficient or conflicting.
3. Retest the same fixed evaluation set after each pipeline change.
4. Evaluate retrieval on a larger and less controlled corpus before drawing conclusions about retriever quality.
5. Preserve separate correctness, retrieval, faithfulness, and failure metrics; a single aggregate score would hide important failure modes.

## Limitations

- only 20 evaluation questions
- deliberately small controlled retrieval corpus
- one small generator model
- one chunking strategy (300 characters, 50-character overlap)
- top-3 retrieval only
- ChromaDB default embedding setup
- manual rubric and failure judgments
- no inter-rater agreement measurement
- generation behavior can vary with decoding and library configuration

The results are findings from this experiment, not benchmark claims about SQuAD v2, Qwen, ChromaDB, or RAG systems generally.

## Skills demonstrated

- LLM evaluation design
- fixed evaluation-set construction
- Hugging Face inference and datasets
- RAG pipeline construction
- overlapping chunking
- ChromaDB semantic retrieval
- Recall@3 and MRR
- rubric-based answer evaluation
- hallucination and failure taxonomy
- faithfulness/groundedness evaluation
- separating retrieval failures from generation failures
- actionable evaluation reporting
