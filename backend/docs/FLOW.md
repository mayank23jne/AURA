# AURA Platform Flow Documentation

## Audit Workflow

The primary workflow in AURA is the compliance audit workflow.

### High-Level Flow

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐
│  Start  │────>│  Policy  │────>│  Test   │────>│  Audit   │
│         │     │Selection │     │Generate │     │ Execute  │
└─────────┘     └──────────┘     └─────────┘     └──────────┘
                                                       │
┌─────────┐     ┌──────────┐     ┌─────────┐           │
│   End   │<────│  Report  │<────│ Analyze │<──────────┘
│         │     │ Generate │     │ Results │
└─────────┘     └──────────┘     └─────────┘
```

### Detailed Audit Flow

```
                    API Request
                         │
                         ▼
              ┌─────────────────────┐
              │  Orchestrator Agent │
              │    (Coordinator)    │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌─────────┐
    │ Policy  │    │ Testing  │    │Knowledge│
    │ Agent   │    │  Agent   │    │  Base   │
    └────┬────┘    └────┬─────┘    └────┬────┘
         │              │               │
         │  Policies    │  Test Cases   │  Context
         └──────────────┼───────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │    Audit Agent      │
              │  (Test Execution)   │
              └──────────┬──────────┘
                         │
                    Test Results
                         │
                         ▼
              ┌─────────────────────┐
              │   Analysis Agent    │
              │ (Result Analysis)   │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌─────────┐
    │Learning │    │  Report  │    │Remediat-│
    │ Agent   │    │  Agent   │    │ion Agent│
    └─────────┘    └──────────┘    └─────────┘
         │              │               │
         │ Experience   │ Report        │ Fixes
         └──────────────┴───────────────┘
                        │
                        ▼
                   API Response
```

## Stage Details

### 1. Initialization Stage

**Trigger**: POST /audit API call

**Actions**:
1. Create audit ID
2. Initialize audit state
3. Record start time
4. Validate request parameters

**State Transition**: `initialization` → `policy_selection`

### 2. Policy Selection Stage

**Agent**: Policy Agent

**Actions**:
1. If specific policies requested: validate and load
2. If no policies specified: select all active policies
3. Validate policy compatibility
4. Check for conflicts

**Output**:
```python
{
    "selected_policies": [...],
    "total_rules": 150,
    "categories": ["safety", "fairness", "transparency"]
}
```

**State Transition**: `policy_selection` → `test_generation`

### 3. Test Generation Stage

**Agent**: Testing Agent

**Actions**:
1. For each policy rule:
   - Generate standard test cases
   - Generate adversarial test cases
   - Generate edge case tests
2. Prioritize tests by severity
3. Deduplicate similar tests

**Test Types**:
- **Standard**: Normal usage scenarios
- **Adversarial**: Attempt to bypass rules
- **Edge Case**: Boundary conditions
- **Synthetic**: LLM-generated scenarios

**Output**:
```python
{
    "test_cases": [...],
    "total_tests": 100,
    "by_type": {
        "standard": 40,
        "adversarial": 30,
        "edge_case": 20,
        "synthetic": 10
    }
}
```

**State Transition**: `test_generation` → `audit_execution`

### 4. Audit Execution Stage

**Agent**: Audit Agent

**Actions**:
1. For each test case:
   - Prepare prompt
   - Query target model
   - Evaluate response
   - Record result
2. Track progress
3. Handle failures/retries

**Execution Pattern**:
```python
for test_case in test_cases:
    # Prepare test
    prompt = prepare_prompt(test_case)

    # Execute against model
    response = await target_model.invoke(prompt)

    # Evaluate compliance
    result = await evaluate_response(
        response,
        test_case.expected_behavior
    )

    # Record result
    results.append(result)
```

**Output**:
```python
{
    "audit_id": "audit_123",
    "results": [
        {
            "test_id": "tc1",
            "passed": True,
            "score": 0.95,
            "response": "...",
            "evaluation": "..."
        }
    ],
    "compliance_score": 0.85
}
```

**State Transition**: `audit_execution` → `analysis`

### 5. Analysis Stage

**Agent**: Analysis Agent

**Actions**:
1. Aggregate results by policy
2. Identify patterns in failures
3. Calculate risk scores
4. Generate findings
5. Prioritize issues

**Analysis Types**:
- Statistical analysis
- Pattern recognition
- Trend analysis
- Correlation detection

**Output**:
```python
{
    "analysis_id": "analysis_123",
    "findings": [
        {
            "type": "pattern",
            "severity": "high",
            "description": "Consistent failures in safety tests",
            "affected_policies": ["safety-001"],
            "confidence": 0.92
        }
    ],
    "risk_assessment": {
        "overall_risk": "medium",
        "risk_areas": ["content_safety"]
    }
}
```

**State Transition**: `analysis` → `reporting`

### 6. Reporting Stage

**Agent**: Report Agent

**Actions**:
1. Compile all results
2. Generate executive summary
3. Create detailed findings
4. Produce recommendations
5. Format report

**Report Sections**:
- Executive Summary
- Methodology
- Results by Policy
- Findings
- Recommendations
- Appendices

**Output**: Complete audit report (JSON/PDF/HTML)

**State Transition**: `reporting` → `learning`

### 7. Learning Stage

**Agent**: Learning Agent

**Actions**:
1. Extract experiences from audit
2. Update knowledge base
3. Refine agent strategies
4. Store patterns for future use

**Learning Types**:
- Successful test patterns
- Failure patterns
- Effective prompts
- Policy improvements

**State Transition**: `learning` → `complete`

## Message Flow

### Inter-Agent Communication

```
┌──────────────┐                    ┌──────────────┐
│    Agent A   │                    │    Agent B   │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │  ┌─────────────────────┐          │
       └─>│    Message Bus      │<─────────┘
          │  (Publish/Subscribe) │
          └─────────────────────┘
```

### Message Lifecycle

1. **Creation**: Agent creates message
2. **Publication**: Message sent to bus
3. **Routing**: Bus routes to target queue
4. **Delivery**: Target agent receives
5. **Processing**: Agent processes message
6. **Response**: Optional response sent

### Priority Handling

Messages are processed by priority (1-10):
- 1-3: Low priority (background tasks)
- 4-6: Normal priority (standard operations)
- 7-8: High priority (important tasks)
- 9-10: Critical (urgent/alerts)

## Event Stream

### Event Types

```python
AUDIT_STARTED = "audit.started"
AUDIT_COMPLETED = "audit.completed"
TEST_EXECUTED = "test.executed"
POLICY_GENERATED = "policy.generated"
FINDING_DETECTED = "finding.detected"
AGENT_ERROR = "agent.error"
```

### Event Flow

```
┌─────────┐     ┌───────────┐     ┌───────────┐
│  Agent  │────>│   Event   │────>│Subscribers│
│         │     │  Stream   │     │           │
└─────────┘     └───────────┘     └───────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Event Log  │
              │  (Persist)  │
              └─────────────┘
```

## Knowledge Flow

### Knowledge Creation

```
┌─────────┐     ┌───────────┐     ┌───────────┐
│ Learning│────>│ Knowledge │────>│  Vector   │
│  Agent  │     │   Item    │     │ Embedding │
└─────────┘     └───────────┘     └───────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Knowledge  │
              │    Base     │
              └─────────────┘
```

### Knowledge Retrieval

```
┌─────────┐     ┌───────────┐     ┌───────────┐
│  Agent  │────>│   Query   │────>│  Semantic │
│         │     │           │     │  Search   │
└─────────┘     └───────────┘     └───────────┘
                                       │
                     ┌─────────────────┘
                     ▼
              ┌─────────────┐
              │   Ranked    │
              │   Results   │
              └─────────────┘
```

## Scheduled Audit Flow

### Scheduler Operation

```
┌─────────┐     ┌───────────┐     ┌───────────┐
│  Cron   │────>│ Scheduler │────>│  Trigger  │
│ Trigger │     │           │     │   Audit   │
└─────────┘     └───────────┘     └───────────┘
                     │
                     ▼
              ┌─────────────┐
              │ Risk-based  │
              │Prioritization│
              └─────────────┘
```

### Risk-Based Scheduling

Models are prioritized based on:
- Last audit date
- Risk score
- Regulatory requirements
- Usage patterns

## Error Handling Flow

### Retry Pattern

```
┌─────────┐     ┌───────────┐     ┌───────────┐
│  Task   │────>│  Execute  │────>│  Success  │
│         │     │           │     │           │
└─────────┘     └─────┬─────┘     └───────────┘
                      │
                   Failure
                      │
                      ▼
              ┌─────────────┐
              │   Retry     │
              │  (3 times)  │
              └──────┬──────┘
                     │
            Still Failing
                     │
                     ▼
              ┌─────────────┐
              │Dead Letter  │
              │   Queue     │
              └─────────────┘
```

### Circuit Breaker

```
┌─────────┐     ┌───────────┐     ┌───────────┐
│ Request │────>│  Circuit  │────>│  Execute  │
│         │     │  Breaker  │     │           │
└─────────┘     └─────┬─────┘     └───────────┘
                      │
              Too Many Failures
                      │
                      ▼
              ┌─────────────┐
              │   Circuit   │
              │    Open     │
              └─────────────┘
                      │
                Wait Period
                      │
                      ▼
              ┌─────────────┐
              │  Half-Open  │
              │  (Test)     │
              └─────────────┘
```

## Data Flow Summary

```
API Request
    │
    ▼
[Validation] → [Orchestration] → [Execution]
                                      │
    ┌─────────────────────────────────┘
    │
    ▼
[Analysis] → [Reporting] → [Learning]
                               │
    ┌──────────────────────────┘
    │
    ▼
Knowledge Base Update
    │
    ▼
API Response
```
