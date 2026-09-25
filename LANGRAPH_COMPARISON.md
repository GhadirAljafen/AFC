# LangGraph vs. Current Pipeline Implementation: Comparative Analysis

## Overview

This document compares the current procedural implementation of the fact-checking pipeline (`src/factcheck_agent/pipeline.py`) with a potential LangGraph-based approach. The analysis focuses on four key areas: state management, control flow, cyclic capabilities, and persistence/human-in-the-loop support.

---

## 1. State Management: Local Variables vs. TypedDict State

### Current Implementation (Procedural)

```python
# Local variables passed between steps
all_evidence: List[EvidenceSnippet] = []
evaluation_result = None
normalized_claim = self.claim_understanding.normalize_claim(claim)
final_evidence = evaluation_result.selected_evidence if evaluation_result else all_evidence
```

**Characteristics:**
- **Scoped to method**: State exists only within the `process()` method
- **Manual passing**: You explicitly pass variables between steps
- **No persistence**: State is lost when the method completes
- **Hard to inspect**: Difficult to debug mid-execution without breakpoints

### LangGraph Approach (TypedDict State)

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class FactCheckState(TypedDict):
    claim: Claim
    normalized_claim: Optional[Claim]
    all_evidence: List[EvidenceSnippet]
    evaluation_result: Optional[EvidenceEvaluationResult]
    iteration_count: int
    final_evidence: List[EvidenceSnippet]
    verdict: Optional[FactCheckVerdict]
    explanation: Optional[str]
    # ... any other state you need
```

**Advantages:**
- **Explicit schema**: All state is defined in one place with types
- **Automatic passing**: State flows automatically between nodes
- **Checkpointing support**: Can save/restore state at any point
- **Easier debugging**: Inspect state at any checkpoint
- **Type safety**: TypedDict provides runtime type checking

**Trade-offs:**
- **More upfront structure**: Need to define state schema
- **Slightly more verbose**: State updates return partial dicts

---

## 2. Control Flow: For Loop vs. State Graph with Conditional Edges

### Current Implementation (Procedural Loop)

```python
for iteration in range(self.max_iterations):
    candidates = await self.retrieval.retrieve_evidence(normalized_claim)
    selected = self.evidence_selection.select(...)
    all_evidence.extend(new_evidence)
    evaluation_result = await self.evidence_evaluation.evaluate(...)
    
    if evaluation_result.sufficient:
        break
    if not new_evidence:
        break
```

**Characteristics:**
- **Linear execution**: Steps run sequentially in a fixed order
- **Break conditions**: Exit conditions scattered throughout the loop
- **Hard to visualize**: Flow is implicit in code structure
- **Difficult to branch**: Adding alternative paths requires refactoring

### LangGraph Approach (Graph with Conditional Edges)

```python
from langgraph.graph import StateGraph, END

# Build graph
workflow = StateGraph(FactCheckState)

# Add nodes
workflow.add_node("normalize_claim", normalize_claim_node)
workflow.add_node("retrieve_evidence", retrieve_evidence_node)
workflow.add_node("select_evidence", select_evidence_node)
workflow.add_node("evaluate_evidence", evaluate_evidence_node)
workflow.add_node("generate_verdict", generate_verdict_node)
workflow.add_node("generate_explanation", generate_explanation_node)

# Define edges
workflow.set_entry_point("normalize_claim")
workflow.add_edge("normalize_claim", "retrieve_evidence")

# Conditional edge: loop or continue?
def should_continue_retrieval(state: FactCheckState) -> str:
    if state["evaluation_result"] and state["evaluation_result"].sufficient:
        return "sufficient"
    if state["iteration_count"] >= max_iterations:
        return "max_iterations_reached"
    if not state.get("new_evidence_found", True):
        return "no_new_evidence"
    return "continue_retrieval"

workflow.add_conditional_edges(
    "evaluate_evidence",
    should_continue_retrieval,
    {
        "sufficient": "generate_verdict",
        "max_iterations_reached": "generate_verdict",
        "no_new_evidence": "generate_verdict",
        "continue_retrieval": "retrieve_evidence"  # Loop back
    }
)

workflow.add_edge("generate_verdict", "generate_explanation")
workflow.add_edge("generate_explanation", END)
```

**Advantages:**
- **Visualizable**: Can render graph as a diagram
- **Explicit decisions**: Conditional edges make branching clear
- **Easy to extend**: Add new branches without refactoring core loop
- **Better testability**: Test individual nodes in isolation
- **Natural parallelism**: Can parallelize independent nodes

**Trade-offs:**
- **More setup code**: Graph definition is more verbose
- **Learning curve**: Team needs to understand graph concepts
- **Slight overhead**: More abstraction for simple linear flows

---

## 3. Cyclic Capabilities: Loop Prevention & Backtracking

### Current Implementation

```python
# Manual loop prevention
for iteration in range(self.max_iterations):  # Hard limit
    # ...
    if not new_evidence:  # Break if no progress
        break
```

**Limitations:**
- **No backtracking**: Can't undo previous evidence collection
- **Fixed max iterations**: Hard-coded limit
- **No reset capability**: Can't easily "start over" with different strategy
- **Irrelevant evidence problem**: If `EvidenceEvaluationAgent` determines previous evidence was irrelevant, you can't easily remove it

### LangGraph Approach

```python
# Built-in cycle prevention
from langgraph.checkpoint.memory import MemorySaver

# Option 1: Interrupt after N cycles
workflow = workflow.compile(checkpointer=MemorySaver())
config = {"recursion_limit": max_iterations}  # Hard limit

# Option 2: Conditional backtracking
def should_backtrack(state: FactCheckState) -> str:
    eval_result = state["evaluation_result"]
    if eval_result and eval_result.notes and "irrelevant" in eval_result.notes.lower():
        return "clear_evidence_and_retry"
    return "continue"

workflow.add_conditional_edges(
    "evaluate_evidence",
    should_backtrack,
    {
        "clear_evidence_and_retry": "clear_evidence_node",  # Custom node to reset
        "continue": "generate_verdict"
    }
)

# Option 3: Dynamic iteration limits based on state
def should_continue_retrieval(state: FactCheckState) -> str:
    # Can check multiple conditions
    if state["iteration_count"] >= state.get("dynamic_max_iterations", 3):
        return "stop"
    # Can check if we're making progress
    if state.get("evidence_quality_trend") == "degrading":
        return "stop"
    return "continue"
```

**Advantages:**
- **Built-in recursion limits**: LangGraph enforces limits automatically
- **Backtracking support**: Can implement nodes that clear state and retry
- **State history**: Checkpoint history lets you see how state evolved
- **Flexible stopping**: Dynamic limits based on quality, not just count

**Example Backtracking Node:**

```python
def clear_irrelevant_evidence_node(state: FactCheckState) -> FactCheckState:
    """Remove evidence marked as irrelevant and reset evaluation."""
    # Filter out evidence marked as irrelevant
    relevant_evidence = [
        e for e in state["all_evidence"]
        if not e.metadata.get("marked_irrelevant", False)
    ]
    return {
        **state,
        "all_evidence": relevant_evidence,
        "evaluation_result": None,  # Reset evaluation
        "iteration_count": state["iteration_count"] + 1
    }
```

---

## 4. Persistence & Human-in-the-Loop: Pausing for Verification

### Current Implementation

```python
async def process(self, claim: Claim) -> FactCheckResult:
    # ... all steps run to completion
    verdict = await self.reasoning_and_verdict.decide(...)
    explanation = await self.explanation.generate(...)
    return FactCheckResult(...)
```

**Limitations:**
- **No pause mechanism**: Everything runs to completion
- **Custom async coordination needed**: Would need to implement queue/event system
- **Ephemeral state**: State lost if process crashes
- **Hard to resume**: No built-in way to continue after human review

### LangGraph Approach

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import interrupt

# Persistent checkpointing
checkpointer = SqliteSaver.from_conn_string(":memory:")  # or file path
workflow = workflow.compile(checkpointer=checkpointer)

# Add interrupt before explanation
def generate_verdict_node(state: FactCheckState) -> FactCheckState:
    verdict = await reasoning_agent.decide(state["normalized_claim"], state["final_evidence"])
    return {
        **state,
        "verdict": verdict
    }

def human_verification_node(state: FactCheckState) -> FactCheckState:
    """Pause and wait for human verification."""
    # This automatically creates an interrupt
    interrupt("human_verification_required")
    # Execution pauses here, state is saved
    return state

# In your application code:
config = {"configurable": {"thread_id": "claim-123"}}
result = workflow.invoke(initial_state, config)

# Check if interrupted
if result.get("__interrupt__"):
    # Save thread_id, show verdict to human
    # Later, resume:
    workflow.update_state(
        config,
        {"human_approved": True}  # Add human feedback to state
    )
    # Continue from where it left off
    final_result = workflow.invoke(None, config)  # Resume
```

**Advantages:**
- **Native interrupt support**: Built-in mechanism to pause execution
- **Automatic persistence**: State saved to checkpoint automatically
- **Thread-based execution**: Handle multiple claims in parallel
- **Human feedback injection**: Can add human decisions to state
- **Timeout support**: Auto-continue if no human response

**Human-in-the-Loop Pattern:**

```python
# 1. Run until verdict
workflow.invoke(state, config)  # Stops at interrupt

# 2. Human reviews verdict (in your web UI/CLI)
verdict = get_state_from_checkpoint(config)["verdict"]
human_decision = await get_human_approval(verdict)

# 3. Resume with human feedback
workflow.update_state(config, {
    "human_approved": human_decision["approved"],
    "human_notes": human_decision["notes"]
})

# 4. Continue to explanation (or skip if human rejected)
final_result = workflow.invoke(None, config)
```

---

## Summary Comparison Table

| Aspect | Current (Procedural) | LangGraph |
|--------|---------------------|-----------|
| **State Management** | Local variables, manual passing | TypedDict, automatic flow |
| **Control Flow** | Linear for loop with breaks | Graph with conditional edges |
| **Visualization** | Code reading only | Can render graph diagram |
| **Loop Prevention** | Manual `max_iterations` + break | Built-in recursion limits |
| **Backtracking** | Difficult (would need refactor) | Natural (add backtracking nodes) |
| **Persistence** | None (ephemeral) | Built-in checkpointing |
| **Human-in-Loop** | Custom async queue needed | Native interrupt support |
| **Debugging** | Print statements, breakpoints | Inspect state at any checkpoint |
| **Parallel Execution** | Manual async coordination | Can parallelize independent nodes |
| **Complexity** | Lower (simple Python) | Higher (graph concepts) |
| **Flexibility** | Hard to add branches | Easy to add alternative paths |

---

## Recommendations

### Migrate to LangGraph if you need:
- ✅ **Human-in-the-loop verification** (pause for review)
- ✅ **State persistence/resumability** (survive crashes, resume later)
- ✅ **Complex branching logic** (different retrieval strategies per claim type)
- ✅ **Visual workflow representation** (documentation, debugging)
- ✅ **Better debugging/inspection** (checkpoint inspection tools)

### Stick with current approach if:
- ✅ Flow is simple and linear
- ✅ You don't need persistence
- ✅ You want minimal dependencies
- ✅ Team prefers procedural code
- ✅ Performance is critical (less abstraction overhead)

### For Your Fact-Checking System

**LangGraph is particularly beneficial if you plan to:**
1. Add human verification workflows (common in production fact-checking)
2. Handle complex retrieval strategies (different sources for different claim types)
3. Need to audit/debug the process (checkpoint inspection)
4. Scale to multiple concurrent claims (thread-based execution)

The checkpointing and interrupt features align particularly well with production fact-checking workflows where human oversight is often required.

---

## Next Steps

If you decide to migrate to LangGraph, the migration path would involve:

1. **Define state schema**: Create `FactCheckState` TypedDict
2. **Convert agents to nodes**: Each agent method becomes a node function
3. **Build graph**: Define nodes and edges
4. **Add conditional logic**: Replace break conditions with conditional edges
5. **Add checkpointing**: Configure checkpointer for persistence
6. **Add interrupts**: Insert interrupt nodes where human review is needed
7. **Update CLI/API**: Use graph's invoke/stream methods instead of direct calls

The good news is that your current agent architecture (separate agent classes) maps cleanly to LangGraph nodes, making the migration relatively straightforward.

