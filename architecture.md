# System Architecture Document

## AI-Powered University Student Services Assistant
**Institution:** Delhi Technological University (DTU)  
**Hackathon:** HCLTech Future Ready AI Engineer Hackathon  
<!-- **Core Architectural Philosophy:** *Determinism Where It Matters* -->

---

## 1. High-Level Architecture Diagram

```mermaid
flowchart TD
    User["Student / Client"] --> UI["Streamlit UI (Port 8501)"]
    UI --> API["FastAPI REST Server (Port 8000)"]
    
    subgraph LangGraph ["LangGraph Deterministic Workflow Engine"]
        N1["1. Guard Node<br/>"]
        N2["2. Classify/Route Node<br/>(Intent & Tool Extraction)"]
        N3["3. Retrieve Node<br/>(ChromaDB Semantic Search)"]
        N4["4. Precedence Node<br/>"]
        N5["5. Deterministic Tools Node<br/>(Attendance, Eligibility)"]
        N6["6. Synthesize Node<br/>(Grounded Answer + Citations R2)"]
        N7["7. Audit Node<br/>(Execution Trace Persistence R10)"]

        N1 -->|Pass| N2
        N1 -->|Refuse R7/R8| N7
        N2 -->|Policy Fact / Procedure| N3
        N2 -->|Personal / Eligibility / What-If| N5
        N2 -->|Ambiguous / Out-of-Scope| N6
        N5 --> N3
        N3 --> N4
        N4 --> N6
        N6 --> N7
    end
    
    API --> N1
    N7 --> API
    
    N3 <--> Chroma[("ChromaDB Vector Store<br/>(Persistent Disk)")]
    N4 <--> Sources[("Source Register<br/>(source_register.csv)")]
    N5 <--> SQLite[("SQLite Database<br/>(Students, Courses, Attendance, Results)")]
    N5 <--> Rules[("Rule Registry Table<br/>(Thresholds & Source Clauses)")]
    N2 & N6 <--> LLM["LLM Layer<br/>(Ollama / MockLLM / Cloud Fallback)"]
```

---

<!-- ## 2. Why Every Component Exists (Justification)

| Component | Responsibility | Why Needed (Requirement Justification) |
|---|---|---|
| **Streamlit UI** | Interactive web frontend | Provides students and evaluators an intuitive interface to select student identities, change `as_of_date` perspectives, view answer-type badges, verifiable citations, and live ingestion. |
| **FastAPI + Uvicorn** | REST API layer | Exposes standardized endpoints (`/ask`, `/ingest`, `/health`, `/sources`, `/audit/{trace_id}`) with strict Pydantic v2 request/response contracts. |
| **LangGraph** | Single-graph orchestration | Coordinates execution without complex multi-agent overhead. Enforces strict determinism and single-path policy routing. |
| **Guard Node** | Header auth & safety check | Enforces R7 (identity comes ONLY from `X-Student-Id` header) and R8 (strips prompt injection attacks in query text). |
| **Classify Node** | Query intent classification | Routes queries into policy fact, procedure, personal data, eligibility, or what-if simulation using Pydantic JSON schemas with deterministic rule fallback. |
| **ChromaDB** | Persistent vector store | Stores chunked official university documents indexed with sentence-transformers; never re-ingests on restart; supports metadata filtering. |
| **Precedence Engine** | Pure-Python policy resolver | Strictly implements the 5-step Source Precedence Policy (Annex A) resolving applicability, explicit supersession, authority level, recency, and conflicts relative to `as_of_date`. |
| **Deterministic Tools** | Python calculation service | Enforces the Core Principle: LLM NEVER calculates attendance percentages, exam eligibility, CGPA checks, or placement eligibility. All calculations are executed by Python over SQLite. |
| **Rule Registry (SQLite)** | Central parameter store | Never hardcodes thresholds in code. Every threshold (75% attendance, 10% condonation, 40% pass mark, 6.50 CGPA, 1 max backlog) is stored in `rule_registry` linked to a cited source document and clause. |
| **Audit Node** | Full trace persistence | Persists complete execution records (`trace_id`, latencies, retrieved chunks, tools invoked, applied rules, precedence decisions) to disk without exposing raw chain-of-thought. |
| **Mock LLM** | Offline deterministic mock | Ensures the entire application, graph, and API can be thoroughly tested with 100% test coverage without running Ollama or an external cloud service. |

---

## 3. Source Precedence Resolution Flow (Annex A)

```mermaid

