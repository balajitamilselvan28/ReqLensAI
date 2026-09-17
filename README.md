# ReqLens AI

A Human-in-the-Loop Multi-Agent Framework for Intelligent Software Requirements Consistency Analysis.

## Features

### Version 1: SRS Document Ingestion

ReqLens AI includes a lightweight PDF ingestion module for reading Software Requirements Specification (SRS) documents.

#### What PDF Ingestion Does
- Accepts a PDF file as input.
- Extracts text page-by-page using PyMuPDF (`fitz`).
- Preserves the page number for each extracted section of text.
- Gracefully handles empty PDFs, corrupted files, invalid file paths, and pages without text.
- Outputs a structured representation (list of dictionaries containing `page_number` and `text`) ready for the Requirement Extraction Agent.

#### How to Use the Parser
```python
from src.utils.pdf_parser import parse_pdf, PDFParsingError

try:
    pages = parse_pdf("path/to/your/srs_document.pdf")
    for page in pages:
        print(f"Page {page['page_number']}:\n{page['text']}\n")
except FileNotFoundError as e:
    print(f"Error: {e}")
except PDFParsingError as e:
    print(f"Error parsing PDF: {e}")
```

#### How to Run Tests
Ensure you have the virtual environment activated and dependencies installed. Then, run the tests using `pytest`:

```bash
python -m pytest -v
```

### Version 3: Semantic Retrieval Agent

The Semantic Retrieval Agent identifies candidate requirement pairs that are semantically related, generating a focused list for downstream contradiction or dependency analysis without comparing every single requirement against all others (O(N^2) avoidance).

#### Important Distinction
Semantic Retrieval is **strictly a candidate-generation stage**. High similarity scores imply the requirements are related in topic, action, or entities, but do **not** necessarily mean they contradict or depend on one another.

#### Implementation Details
- **Embedding Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` as a lightweight embedding engine suitable for basic CPU environments (Intel i3).
- **Vector Store**: Uses `chromadb` configured with cosine similarity (`hnsw:space: cosine`).
- **Algorithm**: Populates the vector store with structured requirements from Version 2, then queries the store for each requirement. Retrieves the `top_k` matches above a specified `threshold`.
- **Duplicate Prevention**: Excludes self-matches and automatically filters out mirrored pairs (e.g., REQ-001/REQ-002 and REQ-002/REQ-001).
- **Output**: Returns a list of structured `CandidatePair` Pydantic models.

#### How to Use
```python
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import RequirementVectorStore
from src.agents.retrieval import SemanticRetrievalAgent

embed_model = EmbeddingModel("all-MiniLM-L6-v2")
vector_store = RequirementVectorStore(embed_model)
agent = SemanticRetrievalAgent(vector_store, top_k=3, threshold=0.6)

# Provide a list of Requirement objects from Version 2
pairs = agent.generate_candidate_pairs(requirements)
```

### Version 4: Knowledge Representation & Logical Reasoning

Version 4 implements deterministic structuring and rule-based evaluation of requirements to support downstream contradiction detection.

#### Knowledge Representation Agent
- **Entities & Relations**: Transforms flat requirement objects into a graph-like schema with distinct nodes (`Actor`, `Action`, `Object`, `Condition`, `Constraint`, `NumericalValue`) and edges (`performs`, `targets`, `has_condition`, `has_constraint`).
- **Dependency Tracking**: The graph schema organically sets up the architecture for later dependency verification (Version 6).

#### Logical Reasoning Agent
- **Rule-Based Engine**: Evaluates candidate pairs (from Version 3) to uncover logical intersections without LLM overhead.
- **Capabilities**: Detects action overlap, actor permission divergence, numerical constraint overlaps (with unit matching), and condition triggers.
- **Strict Evidence Boundaries**: The reasoning agent generates *evidence* (`ReasoningResult`), such as "has_different_actors: True" or "numerical_comparison: 300 vs 500". **It explicitly does not declare contradictions.** The final contradiction decision is strictly reserved for Version 5.
- **CPU Friendly**: The core reasoning is fully deterministic and requires no LLM, meaning it is highly optimized for lightweight environments like an Intel i3.

### Version 5: Contradiction Detection & Dependency Verification

Version 5 consumes the outputs of Versions 3 and 4 to classify potential inconsistencies and verify them, reducing false positives before passing findings to the Recommendation Agent.

#### Contradiction Detection Agent

**Purpose**: Identify structured inconsistencies between candidate requirement pairs.

**Contradiction Categories**:
- `permission_conflict` — Different actors govern the same action on the same object.
- `numerical_conflict` — Differing numerical values constrain the same action with the same unit.
- `conditional_conflict` — Overlapping actions with differing conditional triggers.
- `redundancy` — Substantially equivalent requirements appearing in different places.
- `logical_conflict` — Other logical inconsistencies surfaced by structured evidence.
- `unknown` — High semantic similarity but insufficient deterministic evidence to classify.

**Confidence Scoring**: Built from additive deterministic weights — actor divergence (+0.50), action overlap (+0.25), numerical overlap (+0.30), conditional implication (+0.15). Scores are clamped to [0, 1].

**Potential vs. Verified Findings**:
- `potential` — Detected by structural rules but confidence is below the verification threshold (0.60).
- `verified` — Confidence meets or exceeds the threshold based on structured evidence.
- `rejected` — Downgraded by the Dependency Verification Agent when evidence is insufficient.

A high semantic similarity score alone **never** auto-verifies a finding.

#### Dependency Verification Agent

**Purpose**: Reduce false positives by reviewing each potential finding against structural dependency evidence.

**Dependency Status**:
- `supported` — Strong structural evidence backs the finding; confidence is boosted.
- `conditional` — The conflict may be scoped or limited by conditions; confidence is reduced.
- `unrelated` — No structural dependency confirms the conflict; finding is rejected.
- `insufficient_information` — Evidence is inconclusive; finding remains `potential`.

**False-Positive Reduction**: Conditional conflicts and redundancies are always downgraded before finalizing, ensuring that only structurally well-supported findings reach `verified` status.

**No SRS Modification**: Neither agent modifies, rewrites, or deletes any requirement. That step belongs to later milestones.

## Version 6: Recommendations, Verification, and Human Review

Version 6 adds deterministic recommendation templates, optional LLM-enhanced recommendation text, automated verification, and structured human feedback.

- Recommendations preserve the original finding and evidence.
- The system never automatically rewrites the source SRS.
- Analysts can approve, modify, or reject recommendations through `HumanFeedback`.
- Verification is conservative and flags recommendations that require human review.

## Version 7.1: LangGraph Workflow

The workflow in `src/graph/workflow.py` orchestrates the existing stages in order:

```text
ingest -> extract -> retrieve -> represent -> reason -> detect
    -> dependency -> recommend -> verify -> human_feedback -> report
```

The workflow carries structured state through all stages and records stage errors in `workflow_error`. It uses a deterministic extraction fallback when no LLM is configured. The V2 `RequirementExtractionAgent` remains available for callers that provide a structured-output chat model.

## Version 7.2: Streamlit UI

Run the user interface with:

```bash
streamlit run ui/streamlit_app.py
```

The UI accepts an SRS PDF, parses it with the existing PDF parser, invokes the LangGraph workflow, displays requirements and analysis results, and collects human approval, modification, or rejection decisions. The original SRS is displayed read-only and is never automatically changed.

## Version 7.3: Deterministic Evaluation

The independent evaluation module is located in `evaluation/`:

- `ground_truth.json` contains a small manually defined reference dataset.
- `evaluate.py` calculates the requested metrics from supplied predictions and measured timings.
- `tests/test_evaluation.py` covers perfect, partial, empty, duplicate, false-positive, false-negative, and multi-type cases.

The evaluator reports Requirement Extraction Accuracy, Semantic Retrieval Precision, Contradiction Detection Accuracy, precision, recall, F1-score, Recommendation Acceptance Rate, and Average Processing Time. Example values generated from supplied predictions are evaluation demonstrations, not claims about production performance.

## Installation and Testing

Create and activate a Python virtual environment, then install the declared dependencies:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the complete test suite:

```bash
python -m pytest -q
```

The project uses Sentence Transformers with `all-MiniLM-L6-v2`, ChromaDB, LangGraph, PyMuPDF, Pydantic, and Streamlit. LLM provider keys are optional and must be configured outside source control when using an LLM-backed extractor or recommendation path. `.env.example` contains placeholders only.

## Known Limitations

- The deterministic workflow fallback identifies requirement lines but does not infer all actor, action, object, condition, or numerical fields. Consequently, a no-LLM run can complete successfully while producing no contradiction findings.
- Some contradiction and dependency logic is heuristic and uses exact or normalized string comparisons.
- Semantic retrieval requires the Sentence Transformers model to be installed and available locally or downloadable in the runtime environment.
- The checked-in `venv` directory is not a dependency artifact or deployment environment; recreate it from `requirements.txt` when needed.
- Evaluation infrastructure is present, but no empirical accuracy or production-performance claim is made by this repository alone.

