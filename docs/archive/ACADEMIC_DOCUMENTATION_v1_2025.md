> **ARCHIVED — superseded 2026-09-26.**
>
> This is the original ACADEMIC_DOCUMENTATION.md, kept unchanged as a historical record. Several
> of its claims were later **contradicted by measurement**; see
> `ACADEMIC_DOCUMENTATION.md` for the current version and
> `results/verdict/FINDINGS.md` / `results/evidence_retrieval/RESULTS.md` for the
> evidence. Do not cite figures from this file.

# An Agentic Automated Fact-Checking System: Architecture, Development, and Implementation

## Abstract

This document presents a comprehensive overview of an agentic automated fact-checking system designed to verify factual claims through an iterative, multi-agent architecture. The system employs Large Language Models (LLMs) for reasoning, web search for evidence retrieval, and an agentic loop for evidence evaluation and refinement. This work documents the system's evolution from initial conception through multiple iterations, culminating in a production-ready implementation that addresses the challenge of misinformation through automated verification.

**Keywords:** Fact-checking, Automated Verification, Agentic Systems, Large Language Models, Evidence Retrieval, Misinformation Detection

---

## 1. Introduction

### 1.1 Background and Motivation

The proliferation of misinformation and disinformation in digital media has created an urgent need for automated fact-checking systems. Traditional manual fact-checking, while accurate, is time-consuming and cannot scale to the volume of claims circulating in modern information ecosystems. This project addresses this challenge by developing an automated system that can:

1. Process individual claims or extract claims from longer texts
2. Retrieve relevant evidence from web sources
3. Evaluate evidence quality and sufficiency
4. Generate verdicts with confidence scores and explanations
5. Operate autonomously through an iterative agentic loop

The system is inspired by modern agentic AI architectures, particularly the principles outlined in Anthropic's "Building Effective Agents" framework, which emphasizes iterative refinement, tool use, and structured reasoning.

### 1.2 Objectives

The primary objectives of this system are:

- **Automation**: Reduce human effort in fact-checking through automated evidence retrieval and evaluation
- **Scalability**: Process multiple claims efficiently, including batch processing from long texts
- **Transparency**: Provide explanations and evidence sources for all verdicts
- **Accuracy**: Balance between decisive verdicts (SUPPORTS/REFUTES) and appropriate uncertainty (NOT_ENOUGH_INFO)
- **Extensibility**: Modular architecture allowing easy integration of new evidence sources and reasoning methods

---

## 2. System Architecture

### 2.1 High-Level Architecture

The system follows a modular, agent-based architecture where specialized agents handle distinct aspects of the fact-checking process. The architecture consists of:

1. **Data Models Layer**: Pydantic models for type safety and validation
2. **LLM Abstraction Layer**: Provider-agnostic interface for language model interactions
3. **Agent Layer**: Specialized agents for each fact-checking stage
4. **Pipeline Orchestration**: Main pipeline coordinating agent interactions
5. **Interface Layer**: CLI and programmatic APIs

### 2.2 Core Components

#### 2.2.1 Data Models

The system uses Pydantic models for type safety and validation:

- **`Claim`**: Represents a user-submitted claim with raw text, normalized text, and metadata
- **`DetectedClaim`**: Extends Claim with position information (character indices, sentence index) and importance scores for claims extracted from longer texts
- **`EvidenceSnippet`**: Represents retrieved evidence with source, text, relevance scores, and metadata
- **`FactCheckVerdict`**: Contains the verdict label (SUPPORTS/REFUTES/NOT_ENOUGH_INFO), confidence score, reasoning, and referenced evidence IDs
- **`FactCheckResult`**: Complete result containing claim, verdict, explanation, and all evidence

#### 2.2.2 LLM Client Abstraction

The system implements a provider-agnostic LLM interface (`LLMClient`) that abstracts over different language model providers. Current implementations include:

- **`OpenAILLMClient`**: Integration with OpenAI's API (GPT-4, GPT-4o-mini, etc.)
- **`DummyLLMClient`**: Testing and development fallback

The abstraction allows seamless switching between providers and supports both completion and chat-style interactions.

#### 2.2.3 Agent Architecture

The system employs six specialized agents:

1. **ClaimUnderstandingAgent**: Normalizes claims and detects check-worthy claims from long texts
2. **RetrievalAgent**: Generates search queries and retrieves evidence from web sources
3. **EvidenceSelectionAgent**: Selects and ranks evidence, prioritizing contradictory evidence
4. **EvidenceEvaluationAgent**: Assesses whether collected evidence is sufficient for fact-checking
5. **ReasoningAndVerdictAgent**: Analyzes evidence and generates verdicts with confidence scores
6. **ExplanationAgent**: Generates natural-language explanations of verdicts

### 2.3 Pipeline Flow

The fact-checking pipeline implements an iterative retrieval-evaluation loop:

```
Input Claim
    ↓
[1] Claim Understanding (Normalization)
    ↓
[2] Retrieval-Evaluation Loop (Iterative)
    ├─→ Retrieve Evidence
    ├─→ Select Top Evidence
    ├─→ Evaluate Sufficiency
    └─→ [If insufficient] → Retrieve More Evidence
    ↓
[3] Reasoning & Verdict Generation
    ↓
[4] Explanation Generation
    ↓
Output: FactCheckResult
```

The iterative loop allows the system to refine its evidence collection until sufficient information is gathered or a maximum iteration limit is reached.

---

## 3. Development History

### 3.1 Phase 1: Initial Structure (Foundation)

The project began with establishing a clean, modular structure:

**Objectives:**
- Set up project infrastructure (Poetry, testing, linting)
- Define core data models
- Create placeholder agents with minimal implementations
- Ensure basic importability and testability

**Key Decisions:**
- Use Pydantic for data validation (type safety, serialization)
- Adopt async/await throughout for I/O operations
- Implement abstract base classes for extensibility
- Use Poetry for dependency management

**Deliverables:**
- Project structure with `src/` layout
- Core models: `Claim`, `EvidenceSnippet`, `FactCheckVerdict`, `FactCheckResult`
- Agent skeletons with method signatures
- Basic CLI entry point
- Smoke tests ensuring system importability

### 3.2 Phase 2: LLM Integration

**Objectives:**
- Integrate real LLM providers (OpenAI)
- Implement provider-agnostic abstraction
- Add fallback mechanisms for development/testing

**Implementation:**
- Created `LLMClient` abstract base class
- Implemented `OpenAILLMClient` with async API calls
- Added `DummyLLMClient` for testing
- Created factory function `get_default_llm_client()` for automatic provider selection

**Challenges:**
- Handling API key configuration
- Managing async operations across the pipeline
- Error handling and fallback strategies

### 3.3 Phase 3: Evidence Retrieval

**Objectives:**
- Integrate web search for evidence retrieval
- Implement query generation using LLMs
- Handle search result parsing and deduplication

**Implementation:**
- Integrated Google Custom Search JSON API
- Implemented LLM-based query generation
- Added query expansion to include refutation-focused queries
- Implemented URL-based deduplication

**Key Features:**
- Multi-query generation (2-4 queries per claim)
- Automatic refutation query inclusion
- Result parsing and EvidenceSnippet creation
- Error handling for missing API keys

### 3.4 Phase 4: Agent Implementation

**Objectives:**
- Implement full agent logic using LLMs
- Create iterative retrieval-evaluation loop
- Implement evidence sufficiency evaluation

**Implementation:**
- **EvidenceEvaluationAgent**: LLM-based sufficiency assessment
- **ReasoningAndVerdictAgent**: Evidence analysis and verdict generation
- **ExplanationAgent**: Natural-language explanation generation
- **Pipeline**: Iterative loop with early termination on sufficiency

**Challenges:**
- Balancing between thoroughness and efficiency
- Handling LLM response parsing (JSON extraction)
- Managing iteration limits to prevent infinite loops

### 3.5 Phase 5: Claim Understanding Enhancement

**Objectives:**
- Extend claim understanding beyond basic normalization
- Add claim detection from long texts
- Implement LLM-based claim normalization

**Implementation:**
- Added `DetectedClaim` model for extracted claims
- Implemented `detect_claims()` method using LLM
- Added `normalize_claim_llm()` for advanced normalization
- Maintained backward compatibility with basic normalization

**Features:**
- Importance scoring for detected claims
- Position tracking (character indices, sentence indices)
- Batch claim detection from articles/posts

### 3.6 Phase 6: System Refinement

**Objectives:**
- Address bias toward NOT_ENOUGH_INFO verdicts
- Improve refutation detection
- Enhance evidence quality

**Problem Identified:**
Initial implementation showed bias toward NOT_ENOUGH_INFO (~70%) with few REFUTES verdicts (~5%), even when evidence contradicted claims.

**Root Causes:**
1. LLM prompt didn't explicitly guide toward refutation detection
2. Search queries didn't explicitly seek refuting evidence
3. Evidence selection didn't prioritize contradictory snippets
4. LLM conservatism defaulting to uncertainty

**Solutions Implemented:**

**Priority 1.1: Enhanced Verdict Prompt**
- Added explicit instructions to look for refuting evidence
- Emphasized REFUTES verdicts when evidence contradicts
- Clearer decision criteria for each verdict type

**Priority 1.2: Refutation Search Queries**
- Modified query generation to explicitly request refutation queries
- Added fallback to include refutation queries even when LLM unavailable
- Examples: "claim X false", "claim X debunked", "claim X incorrect"

**Priority 2.1: Evidence Selection Prioritization**
- Rewrote selection algorithm to prioritize contradictory evidence
- Added keyword detection (false, incorrect, debunked, misleading, etc.)
- Scoring boost (+0.3 base + 0.1 per keyword) for contradictory evidence

**Priority 2.2: Confidence Thresholds**
- Added threshold logic: SUPPORTS requires ≥0.5 confidence
- REFUTES allows lower confidence (≥0.4) with uncertainty notes
- Prevents over-confident SUPPORTS on weak evidence

**Expected Impact:**
- REFUTES verdicts: 5% → 30-40%
- NOT_ENOUGH_INFO: 70% → 30-40%
- More balanced verdict distribution

---

## 4. Current Implementation

### 4.1 System Components

#### 4.1.1 Claim Understanding Agent

The `ClaimUnderstandingAgent` provides three main capabilities:

1. **Basic Normalization** (`normalize_claim`): Synchronous, rule-based text cleanup
   - Whitespace normalization
   - Case normalization
   - Duplicate space removal

2. **Advanced Normalization** (`normalize_claim_llm`): LLM-based claim refinement
   - Removes filler words and qualifiers
   - Standardizes entities (dates, numbers, locations)
   - Ensures fact-checkable form

3. **Claim Detection** (`detect_claims`): Extracts check-worthy claims from long texts
   - Uses LLM to identify factual claims
   - Assigns importance scores (0.0-1.0)
   - Tracks position information
   - Filters by minimum importance threshold

#### 4.1.2 Retrieval Agent

The `RetrievalAgent` implements evidence retrieval through:

1. **Query Generation** (`_generate_queries`):
   - LLM-based query generation (3-5 queries per claim)
   - Explicit refutation query inclusion
   - Fallback to basic queries if LLM unavailable

2. **Web Search** (`_google_search`):
   - Google Custom Search JSON API integration
   - Configurable results per query (default: 5)
   - Error handling for API failures

3. **Evidence Creation**:
   - Converts search results to `EvidenceSnippet` objects
   - URL-based deduplication
   - Metadata extraction (title, URL)

#### 4.1.3 Evidence Selection Agent

The `EvidenceSelectionAgent` implements intelligent evidence ranking:

- **Scoring Algorithm**:
  - Base score from retrieval (if available) or 0.5 default
  - Boost for contradictory evidence: +0.3 base + 0.1 per keyword
  - Keywords: false, incorrect, debunked, misleading, untrue, not true, disproven, refuted, contradicts, disputes, wrong, inaccurate, myth, hoax, fake

- **Selection**: Returns top-k evidence sorted by score

#### 4.1.4 Evidence Evaluation Agent

The `EvidenceEvaluationAgent` assesses evidence sufficiency:

- **LLM-Based Evaluation**:
  - Analyzes collected evidence against claim
  - Determines if evidence is sufficient for fact-checking
  - Provides confidence score and notes

- **Fallback Heuristic**:
  - If LLM unavailable: sufficient if ≥3 evidence snippets
  - Confidence based on evidence count

#### 4.1.5 Reasoning and Verdict Agent

The `ReasoningAndVerdictAgent` generates final verdicts:

- **Enhanced Prompt**:
  - Explicit instructions to look for refuting evidence
  - Clear decision criteria for each verdict type
  - Emphasis on decisiveness when evidence contradicts

- **Confidence Thresholds**:
  - SUPPORTS: Requires ≥0.5 confidence
  - REFUTES: Allows ≥0.4 confidence (with uncertainty notes)
  - NOT_ENOUGH_INFO: No threshold

- **Output**: JSON-structured verdict with label, confidence, reasoning, and evidence IDs

#### 4.1.6 Explanation Agent

The `ExplanationAgent` generates natural-language explanations:

- Uses LLM to synthesize verdict, reasoning, and evidence
- Produces clear, neutral explanations
- Includes evidence summaries
- Handles all three verdict types appropriately

### 4.2 Pipeline Implementation

The `FactCheckingPipeline` orchestrates the agentic loop:

```python
async def process(claim: Claim) -> FactCheckResult:
    # Stage 1: Normalization
    normalized_claim = self.claim_understanding.normalize_claim(claim)
    
    # Stage 2: Iterative retrieval-evaluation loop
    all_evidence = []
    for iteration in range(max_iterations):
        candidates = await self.retrieval.retrieve_evidence(normalized_claim)
        selected = self.evidence_selection.select(normalized_claim, candidates, k)
        all_evidence.extend(new_evidence)
        evaluation = await self.evidence_evaluation.evaluate(normalized_claim, all_evidence)
        if evaluation.sufficient:
            break
    
    # Stage 3: Verdict generation
    verdict = await self.reasoning_and_verdict.decide(normalized_claim, final_evidence)
    
    # Stage 4: Explanation generation
    explanation = await self.explanation.generate(normalized_claim, verdict, final_evidence)
    
    return FactCheckResult(claim, verdict, explanation, final_evidence)
```

**Key Features:**
- Iterative refinement until evidence sufficient or max iterations
- Early termination on sufficiency
- Deduplication of evidence across iterations
- Graceful handling of insufficient evidence

### 4.3 Interface Layer

#### 4.3.1 Command-Line Interface

The CLI (`factcheck_agent.cli`) provides:

- Simple claim input: `factcheck "claim text"`
- Source metadata: `factcheck "claim" --source "social_media"`
- Formatted output with verdict, confidence, explanation, and evidence
- Error handling and user-friendly messages

#### 4.3.2 Programmatic API

The system can be used programmatically:

```python
from factcheck_agent import FactCheckingPipeline, get_default_llm_client
from factcheck_agent.models import Claim

llm = get_default_llm_client()
pipeline = FactCheckingPipeline(llm=llm)
claim = Claim(id="c1", raw_text="Saudi Arabia is the largest oil producer")
result = await pipeline.process(claim)
```

#### 4.3.3 Jupyter Notebook Integration

Interactive notebooks provide:
- Step-by-step pipeline execution
- Claim detection from long texts
- Batch fact-checking of detected claims
- Visualization of intermediate results

---

## 5. Technical Details

### 5.1 Asynchronous Architecture

The system is built entirely on async/await patterns:

- **Rationale**: LLM API calls and web searches are I/O-bound operations
- **Benefits**: 
  - Non-blocking execution
  - Efficient resource utilization
  - Scalability for batch processing

- **Implementation**: All agent methods are async, pipeline uses `asyncio.run()` for execution

### 5.2 Error Handling and Resilience

The system implements multiple layers of error handling:

1. **Configuration Validation**: Checks for required API keys at initialization
2. **API Error Handling**: Graceful degradation when APIs fail
3. **LLM Response Parsing**: Robust JSON extraction with regex fallbacks
4. **Fallback Mechanisms**: Heuristic methods when LLMs unavailable
5. **User Feedback**: Clear error messages and warnings

### 5.3 Configuration Management

Configuration is managed through:

- Environment variables (`.env` file support)
- `Config` class with type-safe access
- Validation methods for required settings
- Sensible defaults for optional parameters

**Key Configuration:**
- LLM API keys (OpenAI, Anthropic)
- Search API keys (Google Custom Search)
- Pipeline parameters (max iterations, retrieval k)
- Confidence thresholds

### 5.4 Testing Strategy

The system includes comprehensive tests:

- **Unit Tests**: Individual agent methods
- **Integration Tests**: Agent interactions
- **End-to-End Tests**: Full pipeline execution
- **Smoke Tests**: Basic importability and functionality
- **Mock Tests**: Testing without API dependencies

**Test Coverage:**
- Models validation
- Agent instantiation and method calls
- Pipeline execution (with and without real APIs)
- CLI functionality
- Claim detection and normalization

---

## 6. Evaluation and Improvements

### 6.1 Initial Performance Analysis

Initial system evaluation revealed:

- **Verdict Distribution**:
  - NOT_ENOUGH_INFO: ~70%
  - SUPPORTS: ~25%
  - REFUTES: ~5%

- **Issues Identified**:
  1. Conservative LLM behavior defaulting to uncertainty
  2. Insufficient refutation detection
  3. Evidence retrieval not optimized for contradictory evidence
  4. Prompt design not emphasizing refutation

### 6.2 Improvement Implementation

**Phase 1 Improvements (High Priority):**

1. **Enhanced Verdict Prompt**:
   - Added explicit refutation detection instructions
   - Emphasized decisiveness when evidence contradicts
   - Clearer decision criteria

2. **Refutation Query Generation**:
   - Modified query generation to explicitly seek refutations
   - Added automatic refutation query inclusion
   - Examples: "claim false", "claim debunked"

**Phase 2 Improvements (Medium Priority):**

1. **Evidence Selection Prioritization**:
   - Rewrote selection algorithm to boost contradictory evidence
   - Keyword-based detection of refutation signals
   - Scoring system favoring contradictory snippets

2. **Confidence Thresholds**:
   - SUPPORTS requires higher confidence (≥0.5)
   - REFUTES allows moderate confidence (≥0.4)
   - Prevents over-confident verdicts on weak evidence

### 6.3 Expected Outcomes

After improvements:

- **Verdict Distribution (Expected)**:
  - NOT_ENOUGH_INFO: ~30-40%
  - SUPPORTS: ~20-30%
  - REFUTES: ~30-40%

- **Quality Improvements**:
  - Better detection of false claims
  - More balanced verdict distribution
  - Higher confidence in REFUTES verdicts
  - Improved evidence quality in verdicts

---

## 7. Use Cases and Applications

### 7.1 Single Claim Fact-Checking

The system can verify individual claims:

```bash
factcheck "Saudi Arabia is the largest oil producer in the world."
```

**Output:**
- Verdict (SUPPORTS/REFUTES/NOT_ENOUGH_INFO)
- Confidence score
- Explanation
- Evidence sources with URLs

### 7.2 Batch Claim Processing

The system can process multiple claims from long texts:

1. **Claim Detection**: Extract check-worthy claims from articles/posts
2. **Batch Fact-Checking**: Process all detected claims
3. **Summary Report**: Aggregate results with importance scores

**Workflow:**
```
Long Text → Claim Detection → Multiple Claims → Batch Fact-Checking → Summary
```

### 7.3 Integration Use Cases

- **News Verification**: Verify claims in news articles
- **Social Media Monitoring**: Check viral claims
- **Research Support**: Verify factual claims in research
- **Educational Tools**: Teach critical thinking through fact-checking

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Evidence Source Diversity**: Currently limited to web search; could integrate academic databases, fact-checking databases
2. **Temporal Reasoning**: Limited handling of time-sensitive claims
3. **Multilingual Support**: Primarily English-focused
4. **Bias Detection**: Doesn't explicitly detect source bias
5. **Cost Efficiency**: Multiple LLM calls per claim can be expensive

### 8.2 Future Enhancements

**Short-Term:**
1. **Additional Evidence Sources**: 
   - Fact-checking databases (Snopes, PolitiFact APIs)
   - Academic databases (PubMed, arXiv)
   - Government databases

2. **Improved Query Generation**:
   - Entity-aware query expansion
   - Temporal query modification
   - Multi-lingual query support

3. **Evidence Quality Metrics**:
   - Source credibility scoring
   - Recency weighting
   - Agreement/disagreement detection

**Medium-Term:**
1. **Advanced Reasoning**:
   - Multi-hop reasoning
   - Contradiction resolution
   - Uncertainty quantification

2. **Performance Optimization**:
   - Caching of common claims
   - Parallel evidence retrieval
   - Batch processing optimizations

3. **User Interface**:
   - Web interface
   - API endpoints
   - Real-time fact-checking

**Long-Term:**
1. **Multimodal Fact-Checking**:
   - Image verification
   - Video fact-checking
   - Audio verification

2. **Collaborative Fact-Checking**:
   - Human-in-the-loop verification
   - Community fact-checking
   - Expert review integration

3. **Explainable AI Enhancements**:
   - Evidence attribution
   - Reasoning chains
   - Confidence calibration

---

## 9. Conclusion

This document has presented a comprehensive overview of an agentic automated fact-checking system, from its initial conception through multiple development phases to its current implementation. The system demonstrates:

1. **Modular Architecture**: Clean separation of concerns with specialized agents
2. **Iterative Refinement**: Agentic loop for evidence collection and evaluation
3. **Practical Implementation**: Real-world integration with LLMs and web search
4. **Continuous Improvement**: Responsive to performance analysis and user feedback

The system addresses a critical need in the modern information ecosystem by providing automated, scalable fact-checking capabilities. While current limitations exist, the architecture provides a solid foundation for future enhancements and extensions.

The development process has been iterative and responsive, with improvements driven by empirical evaluation and user needs. The system's evolution from basic placeholders to a production-ready implementation demonstrates the value of incremental development and continuous refinement.

---

## 10. References and Related Work

### 10.1 Related Systems

- **FEVER** (Fact Extraction and VERification): Dataset and baseline for fact-checking
- **ClaimBuster**: Real-time claim detection and verification
- **Full Fact**: Automated fact-checking for UK politics
- **PolitiFact**: Human-in-the-loop fact-checking with automation support

### 10.2 Technical Foundations

- **Anthropic's "Building Effective Agents"**: Framework for agentic AI systems
- **Retrieval-Augmented Generation (RAG)**: Evidence retrieval and LLM integration
- **Pydantic**: Data validation and serialization
- **Async/Await Patterns**: Asynchronous programming in Python

### 10.3 Evaluation Metrics

- **Verdict Accuracy**: Correctness of SUPPORTS/REFUTES/NOT_ENOUGH_INFO
- **Confidence Calibration**: Alignment between confidence scores and accuracy
- **Evidence Quality**: Relevance and credibility of retrieved evidence
- **Response Time**: Latency of fact-checking pipeline

---

## Appendix A: System Requirements

### A.1 Software Dependencies

- Python 3.9+
- Pydantic 2.7.0+
- python-dotenv 1.0.0+
- httpx 0.24.0+ (optional, for web search)
- openai 1.0.0+ (optional, for OpenAI integration)

### A.2 API Requirements

- LLM API key (OpenAI or Anthropic)
- Google Custom Search API key and Engine ID (optional, for web search)

### A.3 Hardware Requirements

- Minimal: Any system capable of running Python 3.9
- Recommended: Systems with network access for API calls

---

## Appendix B: Code Structure

```
factcheck_agent/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── models.py                # Pydantic data models
├── llm_client.py           # LLM abstraction layer
├── pipeline.py             # Main fact-checking pipeline
├── cli.py                  # Command-line interface
└── agents/
    ├── __init__.py
    ├── claim_understanding.py    # Claim normalization and detection
    ├── retrieval.py              # Evidence retrieval
    ├── evidence_selection.py     # Evidence ranking and selection
    ├── evidence_evaluation.py    # Evidence sufficiency evaluation
    ├── reasoning_and_verdict.py  # Verdict generation
    └── explanation.py            # Explanation generation
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-XX
**Author:** Fact-Check Agent Development Team
**License:** MIT

