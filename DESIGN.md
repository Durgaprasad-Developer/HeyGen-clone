# DESIGN.md (Master Multi-Phase Implementation Plan)

## Rules of Engagement for Antigravity Agents

> ### CRITICAL: ANTI-HALLUCINATION GUARDRAILS
> 
> 
> 1. **Phase Lock:** You are strictly forbidden from writing code for a future phase until the current phase passes 100% of its verification tests and the user explicitly gives the command: `"Proceed to Phase X"`.
> 2. **Physical Workspace Verification:** Before and after writing any code, execute a directory sweep (`ls -R`) to verify files are physically written to the actual workspace directory—do not assume a file exists just because it was in your planning context.
> 3. **Executable Test Requirements:** Every phase must include an executable test file (`test_phase_X.py`). If the tests do not execute or if they use hardcoded mock outputs that bypass the actual function logic, the phase is considered a failure.
> 4. **No Placeholders:** Never use `# TODO` or left-blank sections in structural files. Write out full, clean, working production blocks.
> 
> 

CRITICAL: POST-PHASE REPORTING & LEARNING PROTOCOLS
Granular Component Breakdown: Immediately after executing a phase's test suite, you must provide a clean bulleted list detailing every single file created or modified, including its exact role in the microservice architecture.

Architectural State & Flow Update: You must explicitly state how data flows through the newly built components (e.g., "The request hits main.py, passes validation via Pydantic, and initializes state in JOBS_DB"). This ensures the user can learn the operational mechanics.

Explicit Dependency Log: List any third-party packages, libraries, or structural modules introduced during the phase (e.g., fastapi, pytest, torch) so dependencies are clearly tracked.

Test Execution Evidence: Present the raw terminal output of the passing test suite. Do not just summarize the results; show the exact execution metrics so the user can verify the assertion paths.

Static Code Review Highlights: Point out 2–3 critical code snippets or mathematical expressions you implemented in that specific phase and briefly explain why they were structured that way (e.g., why a certain clipping bound or validation length was used).

---

## 1. System Context & Workspace Target Architecture

We are building the isolated backend execution framework for a multimodal avatar video synthesis pipeline. The workspace directory layout must look exactly like this upon completion:

```text
heygen-clone-core/
├── api-gateway/
│   ├── app/
│   │   ├── main.py            # Phase 1: API Router & Task Orchestrator
│   │   ├── config.py          # Phase 1: Environment Settings
│   │   ├── database.py        # Phase 1: Light ORM Isolation Layer
│   │   └── models/
│   │       └── generation_job.py
│   └── Dockerfile
│
├── workers/
│   ├── audio_worker/
│   │   ├── worker.py          # Phase 2: Mocked/Stubbed Audio Queue Worker
│   │   └── tts_engine.py      # Phase 2: F5-TTS interface stub
│   │
│   ├── video_worker/
│   │   ├── worker.py          # Phase 3: Video Orchestration Worker
│   │   ├── inference.py       # Phase 3: Generation model interface stub
│   │   └── alignment/
│   │       ├── grpo_trainer.py # Phase 4: Custom GRPO implementation from scratch
│   │       └── reward_funcs.py # Phase 4: Math-driven visual evaluators
│   └── Dockerfile.gpu
│
└── tests/                     # Phase Verification Suites
    ├── test_phase_1.py
    ├── test_phase_2.py
    ├── test_phase_3.py
    └── test_phase_4.py

```

---

## Phase 1: API Ingestion, Job Tracking & Gateway

### Target Goal

Build a lightweight, production-grade FastAPI application that exposes generation endpoints, creates a mock relational database tracking state, and properly handles validation schemas without crashing.

### Target Files to Create/Modify

* `api-gateway/app/config.py`
* `api-gateway/app/main.py`
* `api-gateway/app/database.py`
* `api-gateway/app/models/generation_job.py`
* `tests/test_phase_1.py`

### Internal Implementation Code Blueprint

```python
# api-gateway/app/main.py
import uuid
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(title="Open-HeyGen Ingestion Engine", version="1.0")

class VideoGenerationRequest(BaseModel):
    avatar_id: str = Field(..., min_length=36, max_length=36, description="UUIDv4 of the target avatar")
    script_text: str = Field(..., min_length=1, max_length=5000, description="The textual script to render")
    user_id: str = Field(..., min_length=36, max_length=36, description="UUIDv4 of the triggering user")

# In-memory database tracking engine state for Phase 1 isolation
JOBS_DB = {}

@app.post("/api/v1/generate", status_code=status.HTTP_202_ACCEPTED)
async def trigger_avatar_generation(payload: VideoGenerationRequest):
    job_id = str(uuid.uuid4())
    JOBS_DB[job_id] = {
        "id": job_id,
        "user_id": payload.user_id,
        "avatar_id": payload.avatar_id,
        "script_text": payload.script_text,
        "status": "pending",
        "error_log": None
    }
    return {
        "status": "queued",
        "job_id": job_id,
        "message": "Pipeline initialization successful."
    }

@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Job entity not found")
    return JOBS_DB[job_id]

```

### Agent System Prompt for Phase 1

> "Execute Phase 1. Read the implementation design for `api-gateway/app/main.py`. Build out the complete, functional FastAPI codebase inside `api-gateway/app/`. Next, build `tests/test_phase_1.py` using `pytest` and `fastapi.testclient`. The tests must completely cover: 1) Valid request parsing and tracking validation, 2) Schema validation errors for invalid input types, 3) Successful state returns via GET paths. Execute the tests using a shell execution command. If they pass perfectly, print out the directory structure and wait for human evaluation."

### Verification Test & Expected Output

Run this test suite locally:

```bash
pytest tests/test_phase_1.py -v

```

**Expected Output:**

```text
tests/test_phase_1.py::test_create_job_success PASSED               [ 33%]
tests/test_phase_1.py::test_create_job_validation_error PASSED      [ 66%]
tests/test_phase_1.py::test_get_job_status_not_found PASSED         [100%]
Verification status: Phase 1 Complete. Awaiting User Signoff.

```

---

## Phase 2: Audio Layer Integration & Worker Pipeline

### Target Goal

Establish the asynchronous message broker communication pattern using a local queue engine. Ensure jobs submitted via the API successfully hand processing state off to an background loop designed to hook into voice cloning frameworks like F5-TTS.

### Target Files to Create/Modify

* `workers/audio_worker/worker.py`
* `workers/audio_worker/tts_engine.py`
* `tests/test_phase_2.py`

### Internal Implementation Code Blueprint

```python
# workers/audio_worker/tts_engine.py
import time

class TTSEngineStub:
    """
    Simulates the Flow Matching Transformer mechanism used in F5-TTS
    without crashing system VRAM footprints during local code testing.
    """
    def __init__(self):
        pass

    def clone_and_synthesize(self, text: str, voice_sample_path: str) -> str:
        # Simulating cross-attention alignment delay
        time.sleep(0.1)
        if not text:
            raise ValueError("Script content cannot be empty string context.")
        return f"/mock_storage/audio_chunks/{uuid.uuid4()}.wav"

```

### Agent System Prompt for Phase 2

> "Execute Phase 2. Build the task worker loops inside `workers/audio_worker/`. Interface the processing logic with our `TTSEngineStub`. Write an integration test suite inside `tests/test_phase_2.py` that instantiates the background task loop manually, pushes an active payload, runs the engine stub, and ensures a string path to a `.wav` file is successfully extracted without dropping errors. Run the tests. Once successful, print results and stop."

### Verification Test & Expected Output

```bash
pytest tests/test_phase_2.py -v

```

**Expected Output:**

```text
tests/test_phase_2.py::test_tts_synthesis_success PASSED            [100%]
Verification status: Phase 2 Complete. Queue mechanics validated. Awaiting User Signoff.

```

---

## Phase 3: Video Layer & Lip Sync Orchestration

### Target Goal

Integrate the cross-attention video modeling layout. The engine must read an input audio track descriptor and process visual keys frame-by-frame, tracking spatial vectors without actual model execution crashing local setups.

### Target Files to Create/Modify

* `workers/video_worker/worker.py`
* `workers/video_worker/inference.py`
* `tests/test_phase_3.py`

### Agent System Prompt for Phase 3

> "Execute Phase 3. Create the video-worker framework. Write the structural framework in `workers/video_worker/inference.py` simulating an architecture like EchoMimic which pairs phonetic inputs to viseme matrices. Write code inside `tests/test_phase_3.py` verifying that passing an input string path and reference coordinate vectors produces a simulated frame array with matching matrix dimension shapes. Execute and confirm code path isolation."

### Verification Test & Expected Output

```bash
pytest tests/test_phase_3.py -v

```

**Expected Output:**

```text
tests/test_phase_3.py::test_viseme_mapping_alignment PASSED         [100%]
Verification status: Phase 3 Complete. Video dimensions confirmed. Awaiting User Signoff.

```

---

## Phase 4: From-Scratch GRPO Optimization Training Loop

### Target Goal

Build out the mathematical, raw algorithmic calculation loop for **Group Relative Policy Optimization (GRPO)** completely from scratch using pure PyTorch logic. This module must calculate reward deviations across groups of generated variations without importing third-party RL packages.

### Target Files to Create/Modify

* `workers/video_worker/alignment/grpo_trainer.py`
* `workers/video_worker/alignment/reward_funcs.py`
* `tests/test_phase_4.py`

### Internal Implementation Code Blueprint

```python
# workers/video_worker/alignment/grpo_trainer.py
import torch

class GRPOTrainer:
    def __init__(self, policy_layers, optimizer, epsilon=0.2):
        self.policy = policy_layers
        self.optimizer = optimizer
        self.epsilon = epsilon

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculates advantage scores relative directly to group metrics."""
        if rewards.size(0) < 2:
            return torch.zeros_like(rewards)
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        return (rewards - mean) / std

    def execute_grpo_step(self, old_log_probs: torch.Tensor, current_log_probs: torch.Tensor, rewards: torch.Tensor) -> float:
        """
        Executes policy updates across a structured generation group array.
        Uses the GRPO objective: computes advantages across variants without using a critic.
        """
        self.optimizer.zero_grad()
        advantages = self.compute_advantages(rewards)
        
        # Calculate ratio of probabilities: r_t(theta) = pi_theta / pi_old
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Compute clipped surrogate objective components
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        
        loss = -torch.min(surr1, surr2).mean()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

```

### Agent System Prompt for Phase 4

> "Execute Phase 4. Implement the explicit mathematical GRPO code inside `workers/video_worker/alignment/grpo_trainer.py` according to the provided layout blueprint. Ensure it strictly calculates standard deviation adjustments and clips loss constraints properly via PyTorch. Write a deep mathematical validation test inside `tests/test_phase_4.py` that feeds synthetic tensor arrays with randomized low and high score fields into the engine. Verify that the tracking gradients compute properly and return numerical float losses. Run the execution step and display outputs."

### Verification Test & Expected Output

```bash
pytest tests/test_phase_4.py -v

```

**Expected Output:**

```text
tests/test_phase_4.py::test_advantage_normalization PASSED          [ 50%]
tests/test_phase_4.py::test_gradient_descent_step PASSED            [100%]
Verification status: Phase 4 Complete. GRPO Math Engine validated. Core Pipeline Complete.

```

---

## Execution Guide for the User

1. **Initialize Workspace:** Drop the text content above directly into a new folder root as `DESIGN.md`.
2. **First Prompt to Antigravity / Claude:**
> "Read `DESIGN.md`. Review the Anti-Hallucination Guardrails. Execute **Phase 1** completely. When the testing suite passes, print out the verified workspace directory layout and wait for my instruction to proceed."


3. Once Phase 1 is done, test it out yourself. If you're happy, reply: `"Proceed to Phase 2"`. Repeat this cadence until the core engine structure is completely complete!

## Phase 5: Live ML Weights & Media Generation (Option B)
### Target Goal
Replace worker stubs with live, operational model weights for F5-TTS and LivePortrait. Take a real 10-second audio sample and a single portrait image baseline, and generate a real, talking `.mp4` file saved locally.

### Target Files to Modify/Create
* `workers/audio_worker/tts_engine.py` (Swap stub out for real F5-TTS pipeline)
* `workers/video_worker/inference.py` (Swap stub out for real LivePortrait pipeline)
* `run_live_pipeline.py` (A master execution script to run an end-to-end live test)