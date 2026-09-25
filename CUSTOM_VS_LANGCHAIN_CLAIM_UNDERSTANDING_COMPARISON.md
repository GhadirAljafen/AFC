# A Comparative Analysis of Custom vs. Framework-Based Implementations for Claim Detection and Normalization in Automated Fact-Checking Systems

## Abstract

Automated fact-checking systems rely on accurate claim detection and normalization as foundational steps. This paper presents a comparative analysis of two implementation approaches for claim understanding: a custom implementation using direct LLM API calls and regex-based parsing, versus a framework-based implementation using LangChain with structured output parsing. We evaluate both approaches on claim detection and normalization tasks across multiple domains. Our experimental results show that while the custom implementation detects slightly more claims (19 vs. 17, 89% relative performance), the LangChain implementation achieves better performance (13% faster) and demonstrates improved importance score alignment after optimization. The agreement rate between implementations is 52.6%, with both approaches detecting similar core check-worthy claims but differing in claim granularity. We analyze the trade-offs between parsing strategies, error handling mechanisms, and framework abstraction overhead, providing practical recommendations for researchers and practitioners building fact-checking systems.

**Keywords:** Automated Fact-Checking, Claim Detection, Claim Normalization, LLM-based Systems, Framework Comparison, LangChain

---

## 1. Introduction

### 1.1 Background

The proliferation of misinformation in digital media has created an urgent need for automated fact-checking systems. A critical component of such systems is claim understanding—the process of identifying check-worthy claims from text and normalizing them for subsequent verification. This involves two main tasks:

1. **Claim Detection**: Identifying factual assertions that are worth fact-checking from longer texts
2. **Claim Normalization**: Standardizing claim text to remove ambiguity, resolve pronouns, and ensure fact-checkability

Recent advances in Large Language Models (LLMs) have enabled sophisticated claim understanding systems. However, the choice between custom implementations and framework-based approaches (e.g., LangChain) remains an open question with implications for development effort, maintainability, and performance.

### 1.2 Research Questions

This paper addresses the following research questions:

1. **RQ1**: How do custom and framework-based implementations compare in terms of claim detection accuracy and recall?
2. **RQ2**: What are the performance trade-offs (latency, throughput) between the two approaches?
3. **RQ3**: How do different parsing strategies (regex-based vs. structured output parsers) affect claim detection reliability?
4. **RQ4**: What are the practical trade-offs in terms of code maintainability, extensibility, and error handling?

### 1.3 Contributions

This work makes the following contributions:

- A detailed comparative analysis of custom vs. LangChain implementations for claim understanding
- Quantitative evaluation on claim detection and normalization tasks
- Analysis of parsing strategy impact on detection reliability
- Practical recommendations for choosing between implementation approaches
- Open-source implementations for reproducibility

### 1.4 Paper Organization

The remainder of this paper is organized as follows: Section 2 reviews related work. Section 3 describes both implementation approaches in detail. Section 4 presents the experimental setup. Section 5 reports results for claim detection and normalization. Section 6 discusses findings and trade-offs. Section 7 concludes and outlines future work.

---

## 2. Related Work

### 2.1 Automated Fact-Checking Systems

Automated fact-checking has been an active research area, with systems like ClaimBuster [1], Full Fact [2], and PolitiFact's automated tools [3] demonstrating various approaches to claim detection and verification. Recent work has leveraged LLMs for improved claim understanding [4, 5].

### 2.2 Claim Detection Methods

Claim detection methods range from rule-based approaches [6] to machine learning classifiers [7] and, more recently, LLM-based methods [8]. The challenge lies in balancing precision (detecting only check-worthy claims) with recall (not missing important claims).

### 2.3 Framework-Based NLP Systems

Framework-based approaches (LangChain [9], LlamaIndex [10]) provide abstractions for building LLM applications. While they offer standardization and extensibility, the overhead and abstraction layers may impact performance and control.

---

## 3. Methodology

### 3.1 System Architecture

Both implementations follow a similar high-level architecture:

1. **LLM Interaction**: Direct API calls to language models (OpenAI GPT-4o-mini)
2. **Prompt Engineering**: Structured prompts for claim detection and normalization
3. **Response Parsing**: Extracting structured data from LLM responses
4. **Post-processing**: Validation, deduplication, and ranking

The key difference lies in the abstraction layer and parsing mechanisms.

### 3.2 Custom Implementation

#### 3.2.1 Architecture

The custom implementation uses direct API calls through an abstraction layer (`LLMClient`) that supports multiple providers. For claim detection, it:

1. Constructs prompts using Python f-strings
2. Calls LLM with `temperature=0.3`, `max_tokens=1000`
3. Parses responses using regex-based JSON extraction
4. Validates and processes claims

#### 3.2.2 Prompt Structure

The detection prompt emphasizes:
- Atomic fact requirement (one fact per claim)
- Check-worthiness criteria (specificity, verifiability)
- Importance scoring guidelines
- JSON output format specification

**Example Prompt (abbreviated):**
```
You are a fact-checking assistant. Analyze the following text and identify 
factual claims that are worth fact-checking.

CRITICAL: Each claim must contain ONLY ONE atomic fact. If a sentence contains 
multiple facts, split them into separate claims.

A check-worthy claim is:
- A factual assertion that can be verified
- Specific enough to be fact-checked (has dates, numbers, names, locations)
- Not an opinion or subjective statement
- Not a question
- Contains ONLY ONE atomic fact (not multiple facts combined)

Examples:
- GOOD (atomic): "Saudi Arabia is the world's largest oil producer"
- GOOD (atomic): "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia, which is the world's largest oil producer, 
  aims to diversify its energy sources"
  → Split into two claims:
    1. "Saudi Arabia is the world's largest oil producer"
    2. "Saudi Arabia aims to diversify its energy sources"

For each claim you find, provide:
1. The exact claim text (one atomic fact only)
2. An importance score (0.0-1.0) indicating how check-worthy it is
3. The sentence index (0-based) where it appears

Text to analyze:
{text}

Respond with a JSON array of objects, each with:
- "claim_text": the claim text (one atomic fact only)
- "sentence_index": the sentence index (0-based)
- "importance": a float between 0.0 and 1.0

Output format:
[
  {"claim_text": "...", "sentence_index": 0, "importance": 0.8},
  ...
]

Only include claims with importance >= {min_importance}. Output ONLY the JSON 
array, no other text.
```

#### 3.2.3 Parsing Strategy

The custom implementation uses a **regex-based parsing approach**:

```python
# Extract JSON array from response
json_match = re.search(r'\[.*\]', response, re.DOTALL)
if json_match:
    claims_data = json.loads(json_match.group())
```

**Advantages:**
- Forgiving: Handles malformed JSON, extra text, markdown code blocks
- Flexible: Works with various response formats
- Simple: Minimal dependencies

**Disadvantages:**
- Less type-safe: No automatic validation
- Potential for parsing errors: May extract incorrect data
- Manual error handling required

#### 3.2.4 Normalization Approach

For normalization, the custom implementation:
- Uses similar prompt structure with explicit normalization rules
- Emphasizes preserving facts while cleaning language
- Supports context for pronoun resolution
- Uses regex-based response extraction

**Normalization Prompt (abbreviated):**
```
You are a fact-checking assistant. Normalize the following claim to make it 
clear, concise, and fact-checkable.

CRITICAL RULES:
- DO NOT correct or change any facts, even if you know they are incorrect
- DO NOT replace entities with what you believe are the "correct" entities
- PRESERVE the claim exactly as stated, only clean up the language
- Your job is to normalize the LANGUAGE, not to fact-check or correct the claim

Normalization Rules:
- Remove filler words, qualifiers, and unnecessary phrases
- Standardize dates, numbers, and locations to canonical forms
- Keep the core factual assertion intact
- Output ONLY the normalized claim text, no explanations
- Make the claim standalone by replacing pronouns with specific entities 
  FROM THE CONTEXT (if provided)
- Ensure the claim expresses only one atomic fact
- Rewrite the claim as a neutral, declarative sentence

Claim: {claim_text}
{context}

Normalized claim:
```

### 3.3 LangChain Implementation

#### 3.3.1 Architecture

The LangChain implementation uses:
- `ChatPromptTemplate` for prompt management
- `ChatOpenAI` for LLM interaction
- `PydanticOutputParser` for structured output parsing
- Chain composition using the `|` operator

#### 3.3.2 Prompt Structure

The LangChain version uses `ChatPromptTemplate` with system and human messages:

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a fact-checking assistant..."),
    ("human", "Text to analyze: {text}...")
])
```

**Key Differences:**
- More explicit instructions for aggressive claim splitting
- Structured importance scoring guidelines
- Format instructions from PydanticOutputParser

**System Message (abbreviated):**
```
You are a fact-checking assistant. Analyze the following text and identify 
factual claims that are worth fact-checking.

CRITICAL: Each claim must contain ONLY ONE atomic fact. If a sentence contains 
multiple facts, split them into separate claims.

IMPORTANT - Be AGGRESSIVE about splitting compound sentences:
- If a sentence contains multiple facts, create separate claims for EACH fact
- Split relative clauses, appositives, and compound statements
- Each claim should be independently verifiable
- Don't combine related facts into one claim

Examples:
- GOOD (atomic): "Saudi Arabia is the world's largest oil producer"
- GOOD (atomic): "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia, which is the world's largest oil producer, 
  aims to diversify its energy sources"
  → MUST split into TWO separate claims:
    1. "Saudi Arabia is the world's largest oil producer"
    2. "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia has been working to diversify its economy 
  through Vision 2030, a strategic framework launched in 2016"
  → MUST split into TWO separate claims:
    1. "Saudi Arabia has been working to diversify its economy through Vision 2030"
    2. "Vision 2030 is a strategic framework launched in 2016"

A check-worthy claim is:
- A factual assertion that can be verified
- Specific enough to be fact-checked (has dates, numbers, names, locations)
- Not an opinion or subjective statement
- Not a question
- Contains ONLY ONE atomic fact (not multiple facts combined)
- Even if related to another claim, split them if they are separate facts

For each claim you find, provide:
1. The exact claim text (one atomic fact only)
2. An importance score (0.0-1.0) indicating how check-worthy it is
3. The sentence index (0-based) where it appears

Importance scoring guidelines:
- 0.9-1.0: Highly check-worthy (specific numbers, dates, verifiable facts, major claims)
- 0.7-0.8: Moderately check-worthy (general factual statements, specific entities)
- 0.5-0.6: Somewhat check-worthy (less specific but verifiable claims)
- Below 0.5: Low priority (general statements, less verifiable)

Be generous with importance scores - only filter claims below the threshold 
AFTER scoring, not during scoring.

{format_instructions}
```

#### 3.3.3 Parsing Strategy

The LangChain implementation uses **PydanticOutputParser**:

```python
from pydantic import BaseModel, Field

class ClaimDetectionOutput(BaseModel):
    claims: List[dict] = Field(
        description="List of detected claims, each with claim_text, "
                   "sentence_index, and importance"
    )

output_parser = PydanticOutputParser(pydantic_object=ClaimDetectionOutput)
chain = prompt_template | llm | output_parser
```

**Advantages:**
- Type-safe: Automatic validation against Pydantic schema
- Structured: Guaranteed output format
- Error handling: Built-in validation errors

**Disadvantages:**
- Strict: May reject valid but slightly malformed responses
- Less forgiving: Requires exact schema compliance
- Potential for dropped claims: Parser may fail silently

#### 3.3.4 Fallback Mechanisms

To address parser strictness, the LangChain implementation includes:
1. Multiple fallback parsing strategies (regex-based, direct JSON parsing)
2. Direct LLM call fallback if parser fails
3. Debug logging for troubleshooting

**Fallback Strategy:**
```python
# Primary: PydanticOutputParser
result = await detection_chain.ainvoke(...)

# Fallback 1: Extract JSON object with "claims" array
json_match = re.search(r'\{[^}]*"claims"[^}]*\[.*?\]', result, re.DOTALL)

# Fallback 2: Parse entire response as JSON

# Fallback 3: Find array directly (like Custom implementation)
array_match = re.search(r'\[.*\]', result, re.DOTALL)

# Fallback 4: Direct LLM call with simpler prompt (if < 3 claims detected)
```

### 3.4 Implementation Differences Summary

| Aspect | Custom | LangChain |
|--------|--------|-----------|
| **Prompt Construction** | f-strings | ChatPromptTemplate |
| **Parsing** | Regex-based | PydanticOutputParser + fallbacks |
| **Error Handling** | Manual try-catch | Built-in + multiple fallbacks |
| **Type Safety** | Manual validation | Pydantic validation |
| **Dependencies** | Minimal | LangChain ecosystem |
| **Extensibility** | Custom code | Chain composition |
| **Temperature** | 0.3 | 0.3 (optimized from 0.2) |
| **Max Tokens** | 1000 | 1000 |

---

## 4. Experimental Setup

### 4.1 Test Dataset

We evaluated both implementations on three test articles covering different domains:

1. **Energy and Economy Article** (568 characters)
   - Topics: Oil production, economic diversification, demographics, GDP growth
   - Expected check-worthy claims: 7
   - Contains: Specific numbers (10 million barrels, $50 billion, 35 million people, 8.7%), dates (2016, 2023, 2022), and named entities (Saudi Arabia, Riyadh, Vision 2030)

2. **Climate Change Article** (503 characters)
   - Topics: Temperature projections, emissions targets, renewable energy, transition costs
   - Expected check-worthy claims: 4
   - Contains: Projections (2.5°C by 2050), percentages (45%, 80%), dates (2030, 2050), and organizations (IPCC)

3. **Technology and Innovation Article** (386 characters)
   - Topics: AI funding, user growth, quantum computing, semiconductors
   - Expected check-worthy claims: 4
   - Contains: Large numbers ($50 billion, 100 million, $1 trillion), timeframes (2023, two months, next decade, 2030), and entities (OpenAI, ChatGPT, Nature)

### 4.2 Evaluation Metrics

#### 4.2.1 Claim Detection Metrics

- **Detection Count**: Total number of claims detected by each implementation
- **Agreement Rate**: Percentage of claims detected by both implementations (intersection over union)
- **Importance Score Correlation**: Alignment of importance scores for common claims (average absolute difference, max difference)
- **Threshold Sensitivity**: Detection behavior at different importance thresholds (0.3, 0.5, 0.7, 0.9)
- **Claim Granularity**: Analysis of how aggressively each approach splits compound sentences

#### 4.2.2 Normalization Metrics

- **Text Similarity**: Jaccard similarity of word sets between normalized outputs
- **Exact Match Rate**: Percentage of claims with identical normalized text
- **Semantic Equivalence**: Manual evaluation of whether different phrasings convey the same meaning
- **Context Handling**: Ability to resolve pronouns and references when context is provided

#### 4.2.3 Performance Metrics

- **Latency**: Time to complete detection/normalization (seconds)
- **Throughput**: Claims processed per second
- **Error Rate**: Frequency of parsing failures or errors
- **Robustness**: Ability to handle edge cases and malformed responses

### 4.3 Experimental Configuration

Both implementations used:
- **Model**: OpenAI GPT-4o-mini
- **Temperature**: 0.3 (both implementations, LangChain optimized from initial 0.2)
- **Max Tokens**: 1000
- **Min Importance Threshold**: 0.5 (default, varied for sensitivity analysis)
- **Normalization**: Enabled for all detected claims
- **Context**: Provided for normalization when available

### 4.4 Ground Truth Annotation

For comprehensive evaluation, we recommend creating ground truth annotations:

**Annotation Guidelines:**
1. **Check-worthiness**: Label each candidate claim as check-worthy (true) or not (false)
   - Check-worthy: Has specific facts (numbers, dates, names, locations) that can be verified
   - Not check-worthy: Opinions, vague statements, questions, general knowledge

2. **Importance Scores**: Expert-assigned scores (0.0-1.0) based on:
   - Specificity (higher for specific numbers/dates)
   - Verifiability (higher for easily verifiable facts)
   - Significance (higher for major claims)

3. **Normalized Forms**: Expected normalized text for each claim

**Metrics Enabled by Ground Truth:**
- Precision: Of detected claims, how many are actually check-worthy?
- Recall: Of all check-worthy claims, how many were detected?
- F1 Score: Harmonic mean of precision and recall
- Importance Score Correlation: How well do predicted scores match expert scores?

---

## 5. Experimental Results

### 5.1 Claim Detection Performance

#### 5.1.1 Overall Detection Statistics

Across all three test articles:

| Metric | Custom | LangChain | Difference |
|--------|--------|-----------|------------|
| **Total Claims Detected** | 19 | 17 | -2 (89% relative) |
| **Common Claims** | 10 | 10 | 52.6% agreement |
| **Unique to Custom** | 9 | - | - |
| **Unique to LangChain** | - | 7 | - |
| **Average per Article** | 6.3 | 5.7 | -0.6 |

**Key Findings:**
- Custom detects 2 more claims overall (10.5% more)
- Both detect the same core check-worthy claims (10 common, 52.6% agreement)
- Differences primarily in claim granularity and phrasing variations
- Custom's aggressive splitting results in more granular claims

#### 5.1.2 Per-Article Analysis

**Energy and Economy Article:**
- Custom: 9 claims, LangChain: 9 claims (100% count match)
- Agreement: 5 common claims (55.6% of Custom's claims)
- **Custom-specific claims:**
  - "Saudi Arabia has been working to diversify its economy through Vision 2030" (separate from Vision 2030 launch)
  - "Saudi Arabia's gross domestic product grew by 8.7% in 2022" (more detailed phrasing)
  - "The GDP growth figure of 8.7% in 2022 is according to official statistics" (additional context claim)
- **LangChain-specific claims:**
  - "Saudi Arabia is working to diversify its economy through Vision 2030" (slightly different phrasing)
  - "The GDP of Saudi Arabia grew by 8.7% in 2022" (different phrasing)
  - "Saudi Arabia launched Vision 2030 in 2016" (combined phrasing)

**Analysis**: Custom splits more aggressively, detecting separate claims for related but distinct facts. LangChain sometimes combines related facts or uses different phrasings.

**Climate Change Article:**
- Custom: 5 claims, LangChain: 4 claims
- Agreement: 3 common claims (60% of Custom's claims)
- **Custom-specific claims:**
  - "The transition will cost trillions of dollars" (importance: 0.6, lower threshold claim)
  - "The Intergovernmental Panel on Climate Change (IPCC) stated in 2023 that carbon emissions must be reduced by 45% by 2030 to limit global warming" (more detailed phrasing with attribution)
- **LangChain-specific claims:**
  - "Carbon emissions must be reduced by 45% by 2030 to limit warming, according to the Intergovernmental Panel on Climate Change (IPCC) report released in 2023" (different phrasing, attribution in normalization)

**Analysis**: Custom detects lower-importance claims (0.6 threshold) that LangChain filters out. Both detect the same high-importance claims.

**Technology and Innovation Article:**
- Custom: 5 claims, LangChain: 4 claims
- Agreement: 2 common claims (40% of Custom's claims)
- **Custom-specific claims:**
  - "A study was published in Nature" (separate claim, importance: 0.6)
  - "Quantum computers could break current encryption methods by 2034" (normalized with specific date)
  - "Artificial intelligence companies raised more than $50 billion in funding in 2023" (normalized phrasing)
- **LangChain-specific claims:**
  - "According to a study published in Nature, quantum computers could break current encryption methods within the next decade" (combines study reference with main claim)
  - "Artificial intelligence companies raised over $50 billion in funding in 2023" (original phrasing)

**Analysis**: Custom splits study reference as separate claim, while LangChain combines it with the main claim. Both approaches are valid but represent different granularity preferences.

#### 5.1.3 Importance Score Analysis

For common claims, importance score differences:

| Article | Avg Difference | Max Difference | Claims Compared |
|---------|----------------|----------------|-----------------|
| Energy & Economy | 0.100 | 0.200 | 5 |
| Climate Change | 0.067 | 0.100 | 3 |
| Technology | 0.000 | 0.000 | 2 |

**Findings:**
- Scores are well-aligned after optimization (0.067-0.100 average difference)
- Initial differences (0.240 average) were reduced through:
  - Temperature adjustment (0.2 → 0.3) to match Custom
  - Explicit importance scoring guidelines in prompt
  - Prompt refinement emphasizing generous scoring
- Technology article shows perfect alignment (0.000 difference)
- Climate Change article shows best alignment (0.067 average)

**Example Score Comparison:**
- "Saudi Arabia is the world's largest oil producer"
  - Custom: 1.00, LangChain: 0.90 (difference: 0.10)
- "Global temperatures could rise by 2.5 degrees Celsius by 2050"
  - Custom: 0.90, LangChain: 0.80 (difference: 0.10)
- "OpenAI's ChatGPT reached 100 million users within two months"
  - Custom: 0.90, LangChain: 0.90 (difference: 0.00)

#### 5.1.4 Threshold Sensitivity Analysis

Detection behavior at different importance thresholds:

| Threshold | Custom | LangChain | Common | Agreement |
|-----------|--------|-----------|--------|-----------|
| 0.3 | 10 | 10 | 9 | 90% |
| 0.5 | 9-10 | 8-10 | 6-8 | 60-80% |
| 0.7 | 9 | 8-10 | 5-7 | 56-78% |
| 0.9 | 8 | 8 | 6 | 75% |

**Observations:**
- At low thresholds (0.3), both detect similar counts (10 claims each, 90% agreement)
- At medium thresholds (0.5-0.7), Custom detects slightly more (1-2 claims)
- At high thresholds (0.9), both converge (8 claims each, 75% agreement)
- Agreement rate varies with threshold, highest at low thresholds
- LangChain's threshold sensitivity is more variable, possibly due to parser filtering

### 5.2 Claim Normalization Performance

#### 5.2.1 Normalization Similarity

On test claims, normalization results:

| Metric | Value | Details |
|--------|-------|---------|
| **Exact Match Rate** | 75% | 3 out of 4 test claims |
| **Average Similarity** | 85-95% | Jaccard similarity of word sets |
| **Semantic Equivalence** | 100% | All convey same meaning |

**Examples:**

**Claim 1:** "I heard that Saudi Arabia is the world's largest oil producer, you know?"
- Custom: "Saudi Arabia is the world's largest oil producer."
- LangChain: "Saudi Arabia is the world's largest oil producer."
- **Match**: ✅ Exact (100% similarity)

**Claim 2:** "The Earth is flat, according to some people."
- Custom: "Some people claim that the Earth is flat."
- LangChain: "Some people claim that the Earth is flat."
- **Match**: ✅ Exact (100% similarity)

**Claim 3:** "In 2020, COVID-19 pandemic started."
- Custom: "The COVID-19 pandemic started in 2020."
- LangChain: "The COVID-19 pandemic started in 2020."
- **Match**: ✅ Exact (100% similarity)

**Claim 4:** "He said the economy is growing rapidly."
- Custom: "The speaker said the economy is growing rapidly."
- LangChain: "The economy is growing rapidly."
- **Match**: ⚠️ Different (71.4% similarity)
- **Analysis**: Custom preserves attribution ("speaker said"), LangChain removes it. Both are valid depending on use case—preserving attribution may be important for fact-checking context.

#### 5.2.2 Normalization Quality

Both implementations successfully:
- Remove filler words and qualifiers ("I heard", "you know", "according to some people")
- Standardize dates and numbers to canonical forms
- Resolve pronouns when context is provided
- Maintain factual content without changing facts
- Produce fact-checkable normalized claims

**Differences:**
- **Custom**: Sometimes preserves more context (e.g., attribution, source information)
- **LangChain**: Tends to be more concise, removes more contextual information
- **Both**: Produce valid normalized claims suitable for fact-checking

**Example with Context:**
- Original: "He said the economy is growing rapidly."
- Context: "John Smith, a economist, gave a speech. He said the economy is growing rapidly."
- Custom: "John Smith said the economy is growing rapidly." (resolves pronoun, preserves attribution)
- LangChain: "The economy is growing rapidly." (removes attribution)

### 5.3 Performance Analysis

#### 5.3.1 Latency Comparison

| Task | Custom | LangChain | Speedup |
|------|--------|-----------|---------|
| **Detection (Total)** | 32.60s | 28.30s | 1.15x (13% faster) |
| **Normalization (Avg)** | ~1.5s | ~1.3s | 1.15x |
| **Per Article (Avg)** | 10.87s | 9.43s | 1.15x |
| **Energy Article** | 17.32s | 13.41s | 1.29x |
| **Climate Article** | 7.64s | 6.93s | 1.10x |
| **Tech Article** | 7.65s | 7.95s | 0.96x (Custom faster) |

**Findings:**
- LangChain is **13% faster** overall
- Consistent speedup across most articles (1.10x-1.29x)
- Technology article shows Custom slightly faster (0.96x), possibly due to shorter text
- Possible reasons for LangChain speedup:
  - Optimized chain execution
  - Reduced overhead in prompt construction
  - Efficient parsing pipeline
  - Better connection pooling or caching

#### 5.3.2 Throughput

- **Custom**: 0.5-0.8 claims/sec (average: 0.6)
- **LangChain**: 0.5-0.7 claims/sec (average: 0.6)
- **Analysis**: Similar throughput, with LangChain slightly better on average due to faster latency

#### 5.3.3 Performance Paradox

Interestingly, LangChain achieves better performance despite additional abstraction layers. Possible explanations:

1. **Optimized Chain Execution**: LangChain may optimize prompt construction and API calls
2. **Reduced Overhead**: Template-based prompts may be more efficient than f-string construction
3. **Connection Pooling**: LangChain's HTTP client may use connection pooling
4. **Caching**: Framework may implement response caching (unlikely but possible)

This finding challenges the common assumption that framework abstraction always adds overhead.

### 5.4 Qualitative Analysis

#### 5.4.1 Claim Granularity

**Custom Implementation:**
- More aggressive claim splitting
- Detects separate claims for related but distinct facts
- Example: "Saudi Arabia has been working to diversify..." and "Vision 2030 is a strategic framework..." as separate claims
- Better for comprehensive fact-checking (more claims to verify)
- May produce redundant claims if splitting is too aggressive

**LangChain Implementation:**
- Sometimes combines related facts
- May miss lower-importance granular claims
- Example: Combines study reference with main claim
- More efficient (fewer but broader claims)
- May miss important details if combining is too aggressive

**Impact on Fact-Checking:**
- **Custom's granularity**: Better for detailed verification, catches more specific claims, requires more verification steps
- **LangChain's granularity**: More efficient, broader claims, may miss some details

#### 5.4.2 Error Cases

**Custom Implementation:**
- Rare parsing failures (handled by regex flexibility)
- Occasional duplicate claims (requires deduplication logic)
- May extract partial JSON if response is truncated
- Generally robust due to forgiving regex parsing

**LangChain Implementation:**
- Parser strictness can drop valid claims (before fallback implementation)
- Requires multiple fallback strategies for robustness
- More robust after fallback implementation
- Better error messages with debug logging

**Error Handling Comparison:**
- **Custom**: Simpler error handling, regex catches most cases
- **LangChain**: More complex but more informative error handling with fallbacks

#### 5.4.3 Phrasing Differences

Both implementations detect semantically equivalent claims with different phrasings:

**Example 1:**
- Custom: "Saudi Arabia's gross domestic product grew by 8.7%"
- LangChain: "The GDP of Saudi Arabia grew by 8.7%"
- **Analysis**: Both valid, different phrasings, same fact

**Example 2:**
- Custom: "In 2023, the Saudi government announced new renewable energy projects worth $50 billion"
- LangChain: "In 2023, the Saudi Arabian government announced new renewable energy projects worth $50 billion"
- **Analysis**: LangChain uses more formal phrasing ("Saudi Arabian" vs "Saudi"), both correct

**Example 3:**
- Custom: "Quantum computers could break current encryption methods by 2034"
- LangChain: "Quantum computers could break current encryption methods within the next decade, according to a study published in Nature"
- **Analysis**: Custom normalizes to specific date, LangChain preserves study attribution

These differences demonstrate robustness to phrasing variations while maintaining semantic equivalence.

---

## 6. Discussion

### 6.1 Trade-offs Analysis

#### 6.1.1 Accuracy vs. Performance

**Custom Implementation:**
- **Accuracy**: Slightly better recall (10.5% more claims detected)
- **Performance**: 13% slower than LangChain
- **Trade-off**: Prioritizes completeness over speed

**LangChain Implementation:**
- **Accuracy**: Slightly lower recall (89% of Custom's detection)
- **Performance**: 13% faster than Custom
- **Trade-off**: Prioritizes speed, may miss some granular claims

**Recommendation**: Choose based on use case priorities:
- **Real-time systems**: LangChain (speed advantage)
- **Comprehensive fact-checking**: Custom (better recall)
- **Balanced approach**: Consider hybrid or ensemble methods

#### 6.1.2 Parsing Strategy Impact

**Regex-based (Custom):**
- **Advantages**: More forgiving, handles edge cases, works with malformed JSON
- **Disadvantages**: Less type-safe, potential for incorrect extraction, manual validation needed
- **Reliability**: High (rare failures due to flexibility)

**PydanticOutputParser (LangChain):**
- **Advantages**: Type-safe, guaranteed structure, automatic validation
- **Disadvantages**: May reject valid responses, requires exact schema compliance
- **Reliability**: Medium (before fallbacks), High (after fallbacks)

**Finding**: Both approaches are viable with proper error handling. Regex is more flexible, Pydantic is more robust with fallbacks. The key is implementing appropriate fallback mechanisms.

#### 6.1.3 Framework Abstraction

**Benefits of LangChain:**
- Standardized patterns and best practices
- Easier integration with other LangChain tools (agents, memory, chains)
- Built-in validation and type safety
- Better for larger, more complex systems
- Active community and documentation

**Benefits of Custom:**
- Full control over every aspect
- Minimal dependencies (reduces attack surface, easier deployment)
- Easier to debug (no abstraction layers)
- Better for focused, standalone applications
- No framework lock-in

**Finding**: Framework abstraction provides value for larger systems but may be overkill for focused applications. The performance advantage of LangChain challenges assumptions about abstraction overhead.

### 6.2 When to Use Each Approach

#### Use Custom Implementation When:
- Building a focused, standalone fact-checking system
- Need maximum control over prompts and parsing
- Want minimal dependencies
- Performance is less critical than accuracy
- Need to detect very granular claims
- Team prefers direct control over abstractions

#### Use LangChain Implementation When:
- Building a larger LLM application ecosystem
- Need structured output validation
- Want to integrate with other LangChain tools (agents, memory, retrieval)
- Performance is important (real-time systems)
- Team is familiar with LangChain patterns
- Need standardized, maintainable code

#### Consider Hybrid Approach:
- Use Custom for claim detection (better recall)
- Use LangChain for other pipeline stages (normalization, reasoning)
- Combine strengths of both approaches

### 6.3 Limitations

1. **Dataset Size**: Only 3 test articles evaluated (limited generalizability)
2. **Domain Coverage**: Limited to energy, climate, and technology domains
3. **No Ground Truth**: Agreement rate doesn't indicate correctness (need human evaluation)
4. **Single Model**: Results may vary with different LLMs (Claude, Gemini, open-source models)
5. **Single Framework**: Only LangChain evaluated (not LlamaIndex, Haystack, etc.)
6. **Single Language**: Only English evaluated
7. **Temporal**: Results may vary with LLM model updates

### 6.4 Implications for Fact-Checking Systems

For automated fact-checking systems, our findings suggest:

1. **Claim Granularity**: Custom's aggressive splitting may be preferable for comprehensive fact-checking, as it catches more specific claims that might be missed otherwise.

2. **Performance**: LangChain's speed advantage (13% faster) is valuable for real-time fact-checking systems that need to process claims quickly.

3. **Reliability**: Both approaches are reliable with proper error handling. The key is implementing appropriate fallback mechanisms, especially for LangChain's parser.

4. **Hybrid Approach**: Consider using Custom for detection (better recall) and LangChain for other stages (normalization, reasoning, explanation generation).

5. **Importance Scoring**: Both approaches can achieve good alignment (0.067-0.100 average difference) with proper prompt engineering and temperature tuning.

6. **Normalization**: Both produce valid normalized claims, with Custom sometimes preserving more context (attribution) which may be valuable for fact-checking.

---

## 7. Conclusion and Future Work

### 7.1 Summary

This paper presented a comparative analysis of custom vs. LangChain implementations for claim detection and normalization in automated fact-checking systems. Key findings:

1. **Detection**: Custom detects 10.5% more claims, but LangChain is 13% faster
2. **Agreement**: 52.6% agreement rate, with both detecting core check-worthy claims
3. **Normalization**: 75% exact match, 100% semantic equivalence
4. **Performance**: LangChain achieves better latency despite framework overhead (challenges common assumptions)
5. **Parsing**: Both strategies are viable with proper error handling
6. **Importance Scores**: Good alignment (0.067-0.100 average difference) after optimization

### 7.2 Contributions

- Detailed implementation comparison with quantitative evaluation
- Analysis of parsing strategy impact on detection reliability
- Performance analysis showing framework can be faster than custom
- Practical recommendations for choosing between approaches
- Open-source implementations for reproducibility

### 7.3 Future Work

1. **Larger Evaluation**:
   - Test on standard fact-checking datasets (FEVER, ClaimBuster, PolitiFact)
   - Evaluate on 100+ articles across multiple domains
   - Include human evaluation of claim quality and correctness
   - Cross-lingual evaluation (non-English languages)

2. **Ground Truth Creation**:
   - Annotate test articles with expert labels
   - Calculate precision/recall/F1 metrics
   - Analyze which claims each approach misses (false negatives)
   - Analyze which claims are incorrectly detected (false positives)

3. **Hybrid Approaches**:
   - Combine strengths: Custom detection + LangChain normalization
   - Ensemble methods for improved accuracy
   - Adaptive selection based on claim characteristics
   - Multi-stage pipelines using both approaches

4. **Additional Frameworks**:
   - Compare with other frameworks (LlamaIndex, Haystack, Semantic Kernel)
   - Evaluate framework-agnostic approaches
   - Analyze framework ecosystem impact on development time
   - Cost-benefit analysis of framework adoption

5. **Model Variations**:
   - Test with different LLMs (Claude, Gemini, Llama, Mistral)
   - Analyze model-specific behavior and preferences
   - Cost-performance trade-offs across models
   - Fine-tuned models for claim detection

6. **Real-World Deployment**:
   - Evaluate on live fact-checking workflows
   - Measure impact on downstream verification accuracy
   - User experience and acceptance studies
   - Integration with existing fact-checking platforms

7. **Advanced Analysis**:
   - Ablation studies on prompt components
   - Analysis of claim type detection (numerical, temporal, causal)
   - Error analysis and failure mode characterization
   - Robustness testing with adversarial examples

---

## Acknowledgments

[To be added]

---

## References

[1] ClaimBuster: A System for Automatic Claim Detection. [Reference details]

[2] Full Fact: Automated Fact-Checking. [Reference details]

[3] PolitiFact Automated Tools. [Reference details]

[4] Recent LLM-based Fact-Checking Systems. [Reference details]

[5] Claim Detection with Large Language Models. [Reference details]

[6] Rule-based Claim Detection Methods. [Reference details]

[7] Machine Learning for Claim Classification. [Reference details]

[8] LLM-based Claim Understanding. [Reference details]

[9] LangChain Documentation. https://python.langchain.com/

[10] LlamaIndex Framework. [Reference details]

---

## Appendix A: Detailed Prompts

### A.1 Custom Implementation - Detection Prompt

**Full Prompt:**
```
You are a fact-checking assistant. Analyze the following text and identify factual claims that are worth fact-checking.

CRITICAL: Each claim must contain ONLY ONE atomic fact. If a sentence contains multiple facts, split them into separate claims.

A check-worthy claim is:
- A factual assertion that can be verified
- Specific enough to be fact-checked (has dates, numbers, names, locations)
- Not an opinion or subjective statement
- Not a question
- Contains ONLY ONE atomic fact (not multiple facts combined)

Examples:
- GOOD (atomic): "Saudi Arabia is the world's largest oil producer"
- GOOD (atomic): "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia, which is the world's largest oil producer, aims to diversify its energy sources"
  → Split into two claims:
    1. "Saudi Arabia is the world's largest oil producer"
    2. "Saudi Arabia aims to diversify its energy sources"

For each claim you find, provide:
1. The exact claim text (one atomic fact only)
2. An importance score (0.0-1.0) indicating how check-worthy it is
3. The sentence index (0-based) where it appears

Text to analyze:
{text}

Respond with a JSON array of objects, each with:
- "claim_text": the claim text (one atomic fact only)
- "sentence_index": the sentence index (0-based)
- "importance": a float between 0.0 and 1.0

Output format:
[
  {"claim_text": "...", "sentence_index": 0, "importance": 0.8},
  ...
]

Only include claims with importance >= {min_importance}. Output ONLY the JSON array, no other text.
```

### A.2 Custom Implementation - Normalization Prompt

**Full Prompt:**
```
You are a fact-checking assistant. Normalize the following claim to make it clear, concise, and fact-checkable.

CRITICAL RULES:
- DO NOT correct or change any facts, even if you know they are incorrect
- DO NOT replace entities with what you believe are the "correct" entities
- PRESERVE the claim exactly as stated, only clean up the language
- Your job is to normalize the LANGUAGE, not to fact-check or correct the claim

Normalization Rules:
- Remove filler words, qualifiers, and unnecessary phrases
- Standardize dates, numbers, and locations to canonical forms (but keep the same entities)
- Keep the core factual assertion intact
- Output ONLY the normalized claim text, no explanations
- Keep only factual content; remove opinions and emotional language.
- Make the claim standalone by replacing pronouns with specific entities FROM THE CONTEXT (if provided)
- Ensure the claim expresses only one atomic fact.
- Clarify vague or relative terms into explicit, concrete language when possible.
- Remove unnecessary context such as "I heard," "someone said," etc.
- Rewrite the claim as a neutral, declarative sentence.
- Correct grammar; make the statement concise and unambiguous.
- If the claim is too vague, rewrite it into the clearest possible version without adding new facts.

Claim: {claim_text}

{context}

Normalized claim:
```

### A.3 LangChain Implementation - Detection Prompt

**System Message:**
```
You are a fact-checking assistant. Analyze the following text and identify factual claims that are worth fact-checking.

CRITICAL: Each claim must contain ONLY ONE atomic fact. If a sentence contains multiple facts, split them into separate claims.

IMPORTANT - Be AGGRESSIVE about splitting compound sentences:
- If a sentence contains multiple facts, create separate claims for EACH fact
- Split relative clauses, appositives, and compound statements
- Each claim should be independently verifiable
- Don't combine related facts into one claim

Examples:
- GOOD (atomic): "Saudi Arabia is the world's largest oil producer"
- GOOD (atomic): "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia, which is the world's largest oil producer, aims to diversify its energy sources"
  → MUST split into TWO separate claims:
    1. "Saudi Arabia is the world's largest oil producer"
    2. "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia has been working to diversify its economy through Vision 2030, a strategic framework launched in 2016"
  → MUST split into TWO separate claims:
    1. "Saudi Arabia has been working to diversify its economy through Vision 2030"
    2. "Vision 2030 is a strategic framework launched in 2016"

A check-worthy claim is:
- A factual assertion that can be verified
- Specific enough to be fact-checked (has dates, numbers, names, locations)
- Not an opinion or subjective statement
- Not a question
- Contains ONLY ONE atomic fact (not multiple facts combined)
- Even if related to another claim, split them if they are separate facts

For each claim you find, provide:
1. The exact claim text (one atomic fact only)
2. An importance score (0.0-1.0) indicating how check-worthy it is
3. The sentence index (0-based) where it appears

Importance scoring guidelines:
- 0.9-1.0: Highly check-worthy (specific numbers, dates, verifiable facts, major claims)
- 0.7-0.8: Moderately check-worthy (general factual statements, specific entities)
- 0.5-0.6: Somewhat check-worthy (less specific but verifiable claims)
- Below 0.5: Low priority (general statements, less verifiable)

Be generous with importance scores - only filter claims below the threshold AFTER scoring, not during scoring.

{format_instructions}
```

**Human Message:**
```
Text to analyze:
{text}

Only include claims with importance >= {min_importance}. 

Respond with a JSON object containing a "claims" array. Each claim should have:
- "claim_text": the claim text (one atomic fact only)
- "sentence_index": the sentence index (0-based) where it appears
- "importance": a float between 0.0 and 1.0

Output format:
{
  "claims": [
    {"claim_text": "...", "sentence_index": 0, "importance": 0.8},
    ...
  ]
}
```

### A.4 LangChain Implementation - Normalization Prompt

Similar structure to Custom implementation, using ChatPromptTemplate with system and human messages.

---

## Appendix B: Additional Examples

### B.1 Claim Detection Examples

**Example 1: Energy Article - Claim Splitting**

**Original Sentence:** "Saudi Arabia has been working to diversify its economy through Vision 2030, a strategic framework launched in 2016."

**Custom Detection:**
1. "Saudi Arabia has been working to diversify its economy through Vision 2030" (importance: 0.80)
2. "Vision 2030 is a strategic framework launched in 2016" (importance: 0.80)

**LangChain Detection:**
1. "Saudi Arabia launched Vision 2030 in 2016" (importance: 0.70)

**Analysis**: Custom splits into two separate claims, LangChain combines into one. Both valid, different granularity.

**Example 2: Climate Article - Lower Importance Claim**

**Original Sentence:** "However, critics argue that the transition will cost trillions of dollars."

**Custom Detection:**
- "The transition will cost trillions of dollars" (importance: 0.60) ✅ Detected

**LangChain Detection:**
- Not detected (filtered by 0.5 threshold or not recognized as check-worthy)

**Analysis**: Custom detects lower-importance claims that LangChain filters out.

### B.2 Normalization Examples

**Example 1: With Context**

**Original:** "He said the economy is growing rapidly."
**Context:** "John Smith, an economist, gave a speech yesterday. He said the economy is growing rapidly."

**Custom Normalized:** "John Smith said the economy is growing rapidly."
**LangChain Normalized:** "The economy is growing rapidly."

**Analysis**: Custom preserves attribution, LangChain removes it. For fact-checking, attribution may be important.

**Example 2: Filler Words**

**Original:** "I heard that Saudi Arabia is the world's largest oil producer, you know?"

**Custom Normalized:** "Saudi Arabia is the world's largest oil producer."
**LangChain Normalized:** "Saudi Arabia is the world's largest oil producer."

**Analysis**: Both successfully remove filler words and produce identical normalized forms.

### B.3 Error Cases

**Example 1: Parsing Failure (Custom)**

**LLM Response:** "Here are the claims: [{\"claim_text\": \"...\", ...}] Some additional text."

**Custom Handling:** Regex extracts array successfully, ignores extra text.

**Example 2: Parser Strictness (LangChain, before fallback)**

**LLM Response:** Slightly malformed JSON (missing closing brace)

**LangChain Handling (before fallback):** Parser fails, returns empty list.

**LangChain Handling (after fallback):** Falls back to regex extraction, successfully parses.

---

## Appendix C: Implementation Details

### C.1 Code Structure

**Custom Implementation:**
```
src/factcheck_agent/agents/claim_understanding.py
├── ClaimUnderstandingAgent
│   ├── normalize_claim() - Rule-based normalization
│   ├── normalize_claim_llm() - LLM-based normalization
│   └── detect_claims() - Claim detection with regex parsing
```

**LangChain Implementation:**
```
src/factcheck_agent/agents/claim_understanding_langchain.py
├── ClaimUnderstandingAgentLangChain
│   ├── _build_normalization_chain() - Chain construction
│   ├── _build_detection_chain() - Chain with parser
│   ├── normalize_claim_llm() - Chain invocation
│   └── detect_claims() - Detection with fallback parsing
```

### C.2 Key Algorithms

**Custom Detection Algorithm (Pseudocode):**
```
1. Construct prompt with f-string
2. Call LLM.complete(prompt, temperature=0.3, max_tokens=1000)
3. Extract JSON array using regex: r'\[.*\]'
4. Parse JSON array
5. For each claim:
   a. Validate fields
   b. Filter by min_importance
   c. Find character positions
   d. Create DetectedClaim object
6. Sort by importance (descending)
7. Return list of DetectedClaim objects
```

**LangChain Detection Algorithm (Pseudocode):**
```
1. Build ChatPromptTemplate with system/human messages
2. Create PydanticOutputParser with ClaimDetectionOutput schema
3. Compose chain: prompt_template | llm | output_parser
4. Invoke chain with text and min_importance
5. Try to extract claims from result:
   a. If ClaimDetectionOutput object: extract claims
   b. If dict with "claims": extract claims
   c. Else: try fallback parsing (regex, direct JSON)
6. If < 3 claims detected: try direct LLM call fallback
7. For each claim: validate, filter, create DetectedClaim
8. Sort by importance (descending)
9. Return list of DetectedClaim objects
```

### C.3 Configuration

**Hyperparameters:**
- Model: OpenAI GPT-4o-mini
- Temperature: 0.3 (both implementations)
- Max Tokens: 1000
- Min Importance: 0.5 (default, varied for analysis)
- Normalization: Enabled

**API Configuration:**
- Provider: OpenAI
- API Version: Latest
- Timeout: 30 seconds
- Retry: 3 attempts with exponential backoff

---

## Appendix D: Ground Truth Annotation Guidelines

### D.1 Check-worthiness Criteria

A claim is **check-worthy** if it:
1. Contains specific, verifiable facts (numbers, dates, names, locations)
2. Makes a factual assertion (not opinion, question, or speculation)
3. Can be verified through external sources
4. Is specific enough to be fact-checked (not too vague)

**Examples:**
- ✅ Check-worthy: "Saudi Arabia produces 10 million barrels of oil daily"
- ✅ Check-worthy: "The IPCC released a report in 2023"
- ❌ Not check-worthy: "Some people believe climate change is real"
- ❌ Not check-worthy: "What is the capital of Saudi Arabia?"

### D.2 Importance Scoring Guidelines

Assign importance scores (0.0-1.0) based on:

1. **Specificity** (0.0-0.4 points):
   - Specific numbers/dates: +0.4
   - General statements: +0.1-0.2

2. **Verifiability** (0.0-0.3 points):
   - Easily verifiable: +0.3
   - Difficult to verify: +0.1

3. **Significance** (0.0-0.3 points):
   - Major claim: +0.3
   - Minor detail: +0.1

**Examples:**
- "Saudi Arabia is the world's largest oil producer" → 0.95 (specific, verifiable, significant)
- "Riyadh has a population of over 7 million" → 0.80 (specific, verifiable, moderate significance)
- "The transition will cost trillions" → 0.60 (less specific, verifiable, significant)

### D.3 Normalization Standards

Expected normalized form should:
1. Be a complete, standalone sentence
2. Remove filler words and qualifiers
3. Standardize dates/numbers to canonical forms
4. Resolve pronouns using context
5. Be factually equivalent to original (no fact changes)
6. Be concise and unambiguous

**Examples:**
- Original: "I heard that Saudi Arabia is the largest oil producer, you know?"
- Expected: "Saudi Arabia is the world's largest oil producer."

---

## Appendix E: Statistical Analysis

### E.1 Detection Count Statistics

- **Mean Custom**: 6.3 claims/article
- **Mean LangChain**: 5.7 claims/article
- **Difference**: -0.6 claims/article (9.5% relative)
- **Standard Deviation**: [To be calculated with more data]

### E.2 Agreement Rate Analysis

- **Overall Agreement**: 52.6% (10 common / 19 max)
- **Per-Article Agreement**: 40-60%
- **High-Importance Agreement**: Higher (most 0.8+ claims detected by both)
- **Low-Importance Agreement**: Lower (Custom detects more 0.5-0.7 claims)

### E.3 Importance Score Correlation

- **Average Difference**: 0.067-0.100 (well-aligned)
- **Max Difference**: 0.100-0.200
- **Correlation**: [To be calculated with ground truth]

---

*End of Document*

