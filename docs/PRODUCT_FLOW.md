# KAIROS Clinical Admin Edge (CAE) - Product Flow Guide

## Overview

KAIROS CAE is a **100% offline medical documentation system** that automates administrative tasks for clinics. It captures patient-clinician conversations, generates structured medical reports (Compte Rendu), and syncs data to legacy EHR systems - all with mandatory human verification.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           KAIROS CAE SYSTEM                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │ RECEPTIONIST │    │  CLINICIAN   │    │    ADMIN     │                 │
│  │    PORTAL    │    │    PORTAL    │    │    PORTAL    │                 │
│  │              │    │              │    │              │                 │
│  │ - Record     │    │ - Review     │    │ - Monitor    │                 │
│  │ - Transcribe │    │ - Generate   │    │ - Configure  │                 │
│  │ - Tag        │    │ - Verify     │    │ - Manage     │                 │
│  │              │    │ - Sync       │    │              │                 │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                 │
│         │                   │                   │                          │
│         └───────────────────┴───────────────────┘                          │
│                             │                                              │
│                    ┌────────▼────────┐                                     │
│                    │   FastAPI       │                                     │
│                    │   Backend       │                                     │
│                    └────────┬────────┘                                     │
│                             │                                              │
│         ┌───────────────────┼───────────────────┐                          │
│         │                   │                   │                          │
│  ┌──────▼──────┐    ┌───────▼───────┐   ┌──────▼──────┐                   │
│  │   WHISPER   │    │    PARLANT    │   │  DEEPSEEK   │                   │
│  │   (Audio)   │    │    (Agent)    │   │    VL2      │                   │
│  │             │    │               │   │  (Vision)   │                   │
│  │ Transcribe  │    │ Generate CR   │   │ Screen OCR  │                   │
│  │ + Keywords  │    │ + RAG         │   │ + RPA       │                   │
│  └─────────────┘    └───────────────┘   └─────────────┘                   │
│                             │                                              │
│                    ┌────────▼────────┐                                     │
│                    │  LOCAL STORAGE  │                                     │
│                    │  SQLite + Qdrant│                                     │
│                    └─────────────────┘                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete User Flows

### Flow 1: Patient Consultation Recording

**Actor:** Receptionist
**Goal:** Capture and transcribe patient-clinician conversation

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RECEPTIONIST WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. LOGIN                                                           │
│     └─> Open Receptionist Portal                                    │
│         └─> No PIN required (public access)                         │
│                                                                     │
│  2. CREATE SESSION                                                  │
│     ├─> Enter Patient ID (optional)                                 │
│     ├─> Select Language (French/English)                            │
│     └─> Click "Start Session"                                       │
│         └─> System creates session_id                               │
│         └─> WebSocket connection established                        │
│                                                                     │
│  3. RECORD CONVERSATION                                             │
│     ├─> Click "Start Recording"                                     │
│     │   └─> Browser requests microphone permission                  │
│     │   └─> Audio streams to backend via WebSocket                  │
│     │                                                               │
│     ├─> REAL-TIME PROCESSING                                        │
│     │   ├─> Whisper transcribes audio chunks                        │
│     │   ├─> Keywords extracted and highlighted                      │
│     │   ├─> Confidence scores displayed                             │
│     │   └─> Transcript scrolls in real-time                         │
│     │                                                               │
│     ├─> Controls Available:                                         │
│     │   ├─> Pause Recording                                         │
│     │   ├─> Resume Recording                                        │
│     │   └─> End Session                                             │
│     │                                                               │
│     └─> ALTERNATIVE: Upload Pre-recorded Audio                      │
│         └─> Supports WAV, MP3, M4A                                  │
│         └─> Batch transcription                                     │
│                                                                     │
│  4. COMPLETE SESSION                                                │
│     └─> Click "End Session"                                         │
│         └─> Session marked "completed"                              │
│         └─> Full transcript saved to database                       │
│         └─> Available for clinician review                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**API Calls:**
- `POST /session/start` → Create session
- `WS /audio/stream/{session_id}` → Stream audio
- `POST /session/{id}/pause` → Pause
- `POST /session/{id}/resume` → Resume
- `POST /session/{id}/end` → End session

---

### Flow 2: Medical Report Generation & EHR Sync

**Actor:** Clinician
**Goal:** Review transcript, generate Compte Rendu, sync to legacy EHR

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLINICIAN WORKFLOW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. AUTHENTICATE                                                    │
│     └─> Enter Staff PIN (default: 1234)                             │
│         └─> Access granted to protected features                    │
│                                                                     │
│  2. SELECT SESSION                                                  │
│     ├─> View list of completed sessions                             │
│     ├─> Filter by status, date, patient                             │
│     └─> Click session to open                                       │
│                                                                     │
│  3. REVIEW TRANSCRIPT                                               │
│     ├─> Full transcript displayed                                   │
│     ├─> Medical keywords highlighted                                │
│     ├─> Timestamps for each segment                                 │
│     └─> Confidence indicators shown                                 │
│                                                                     │
│  4. GENERATE COMPTE RENDU                                           │
│     └─> Click "Generate Draft"                                      │
│         │                                                           │
│         ├─> PARLANT AGENT PROCESSING:                               │
│         │   ├─> Fetches transcript from database                    │
│         │   ├─> Queries Qdrant for relevant protocols (RAG)         │
│         │   ├─> Builds structured prompt                            │
│         │   ├─> Sends to Ollama (local LLM)                         │
│         │   └─> Parses response into CR structure                   │
│         │                                                           │
│         └─> RESULT DISPLAYED:                                       │
│             ├─> Patient Information                                 │
│             │   ├─> Name, DOB, MRN                                  │
│             │   └─> Consultation date                               │
│             │                                                       │
│             ├─> Medical Sections                                    │
│             │   ├─> Motif de Consultation                           │
│             │   ├─> Anamnese                                        │
│             │   ├─> Antecedents                                     │
│             │   ├─> Allergies                                       │
│             │   ├─> Traitements Actuels                             │
│             │   ├─> Examen Clinique                                 │
│             │   ├─> Examens Complementaires                         │
│             │   ├─> Hypotheses Diagnostiques                        │
│             │   └─> Plan Therapeutique                              │
│             │                                                       │
│             ├─> ALERTS (Color Coded)                                │
│             │   ├─> YELLOW: Missing required fields                 │
│             │   └─> ORANGE: Needs clinical validation               │
│             │                                                       │
│             └─> RAG CITATIONS                                       │
│                 └─> Protocol references with sources                │
│                                                                     │
│  5. HUMAN VERIFICATION (MANDATORY)                                  │
│     ├─> Review each section                                         │
│     ├─> Edit any incorrect information                              │
│     ├─> Address yellow/orange alerts                                │
│     └─> Mark sections as "Verified"                                 │
│                                                                     │
│  6. EHR VISION CAPTURE (Optional)                                   │
│     └─> Click "Capture EHR Screen"                                  │
│         ├─> DeepSeek-VL2 captures legacy EHR window                 │
│         ├─> Extracts current patient data                           │
│         ├─> Shows side-by-side comparison                           │
│         └─> Confidence scores per field                             │
│                                                                     │
│  7. REQUEST RPA SYNC                                                │
│     └─> Click "Sync to EHR"                                         │
│         │                                                           │
│         ├─> VERIFICATION REQUEST CREATED:                           │
│         │   ├─> Preview of all RPA actions shown                    │
│         │   ├─> Field-by-field breakdown                            │
│         │   └─> Estimated coordinates displayed                     │
│         │                                                           │
│         ├─> REVIEW PREVIEW                                          │
│         │   ├─> What will be clicked                                │
│         │   ├─> What will be typed                                  │
│         │   └─> What fields will be modified                        │
│         │                                                           │
│         └─> APPROVE OR REJECT                                       │
│             ├─> Click "Approve" → RPA executes                      │
│             │   ├─> Mouse moves to coordinates                      │
│             │   ├─> Clicks input fields                             │
│             │   ├─> Types verified text                             │
│             │   ├─> Visual verification after each action           │
│             │   └─> Reports success/failure                         │
│             │                                                       │
│             └─> Click "Reject" → Sync cancelled                     │
│                 └─> Audit trail preserved                           │
│                                                                     │
│  8. POST-SYNC VERIFICATION                                          │
│     ├─> System captures EHR screen after typing                     │
│     ├─> Verifies data was entered correctly                         │
│     └─> Reports any discrepancies                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**API Calls:**
- `GET /session/` → List sessions
- `GET /session/{id}` → Get session details
- `GET /agent/draft?session_id={id}` → Generate CR
- `POST /vision/scrape` → Capture EHR screen
- `POST /agent/sync` → Request RPA sync
- `POST /agent/sync/{id}/approve` → Execute RPA
- `POST /agent/sync/{id}/reject` → Cancel sync

---

### Flow 3: System Administration

**Actor:** Administrator
**Goal:** Configure system, manage protocols, monitor health

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ADMIN WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. AUTHENTICATE                                                    │
│     └─> Enter Admin PIN                                             │
│                                                                     │
│  2. SYSTEM STATUS TAB                                               │
│     └─> View health of all services:                                │
│         ├─> Database (SQLite)        ● Ready / ○ Error              │
│         ├─> Ollama (LLM)             ● Ready / ○ Error              │
│         ├─> Parlant (Agent)          ● Ready / ○ Error              │
│         ├─> Whisper (Audio)          ● Ready / ○ Error              │
│         ├─> Vision (DeepSeek-VL2)    ● Ready / ○ Error              │
│         └─> Qdrant (Vector DB)       ● Ready / ○ Error              │
│                                                                     │
│  3. PROTOCOL MANAGEMENT TAB                                         │
│     ├─> SEARCH PROTOCOLS                                            │
│     │   ├─> Enter search query                                      │
│     │   └─> View results with relevance scores                      │
│     │                                                               │
│     ├─> INGEST NEW PROTOCOL                                         │
│     │   ├─> Enter title, content, source                            │
│     │   ├─> Select language (FR/EN)                                 │
│     │   └─> Click "Add Protocol"                                    │
│     │       └─> Embedded and stored in Qdrant                       │
│     │                                                               │
│     └─> CLEAR ALL PROTOCOLS                                         │
│         └─> Confirmation required                                   │
│                                                                     │
│  4. MODEL MANAGEMENT TAB                                            │
│     ├─> VIEW AVAILABLE MODELS                                       │
│     │   └─> List from Ollama with sizes                             │
│     │                                                               │
│     ├─> SELECT ACTIVE MODEL                                         │
│     │   └─> Switch LLM for generation                               │
│     │                                                               │
│     └─> PULL NEW MODEL                                              │
│         ├─> Enter model name (e.g., "mistral")                      │
│         └─> Stream download progress                                │
│                                                                     │
│  5. LEGACY EHR CONFIGURATION (Legacy Mapper)                        │
│     ├─> SELECT EHR WINDOW                                           │
│     │   └─> List available windows                                  │
│     │                                                               │
│     ├─> AUTO-DETECT LAYOUT                                          │
│     │   └─> DeepSeek-VL2 finds input fields                         │
│     │                                                               │
│     ├─> MANUAL CALIBRATION                                          │
│     │   ├─> Click to set field coordinates                          │
│     │   └─> Name each field mapping                                 │
│     │                                                               │
│     └─> SAVE COORDINATE MAP                                         │
│         └─> Stored for future RPA operations                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**API Calls:**
- `GET /admin/status` → System health
- `POST /admin/protocols/search` → Search protocols
- `POST /admin/protocols/ingest` → Add protocols
- `DELETE /admin/protocols/clear` → Clear protocols
- `GET /admin/models` → List models
- `POST /admin/models/select` → Switch model
- `GET /admin/models/pull/{name}` → Pull model
- `POST /vision/calibrate` → Save coordinates
- `POST /vision/detect-layout` → Auto-detect

---

## Data Flow Diagram

```
                    PATIENT CONSULTATION
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    AUDIO CAPTURE                             │
│  Microphone → WebSocket → Whisper → Transcript + Keywords    │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    DOCUMENT GENERATION                       │
│  Transcript + RAG Protocols → Parlant Agent → Compte Rendu   │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    HUMAN VERIFICATION                        │
│  Clinician Reviews → Edits → Approves Each Section           │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    EHR SYNCHRONIZATION                       │
│  Vision Capture → RPA Preview → Approval → Keystroke Inject  │
│                         │                                    │
│                         ▼                                    │
│              Post-Sync Visual Verification                   │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    LEGACY EHR UPDATED
```

---

## Safety & Compliance Features

### Human-in-the-Loop Verification

**ALL automated actions require human approval:**

1. **Document Generation**
   - AI suggests content
   - Clinician must verify each section
   - Missing/flagged fields highlighted

2. **RPA Execution**
   - Preview of all actions shown
   - Staff PIN required for approval
   - Can modify before execution

3. **Post-Action Verification**
   - Screen captured after RPA
   - Visual confirmation of data entry
   - Discrepancies reported

### Administrative Guardrails

The Parlant agent enforces strict boundaries:

```
GUARDRAIL 1: No Medical Diagnosis
├─> If AI detects diagnostic request
└─> Response: "I am an administrative assistant;
    I have flagged this area for your clinical input."

GUARDRAIL 2: Missing Data Alerts
├─> If required field not in transcript
└─> Adds "Missing Field" tag with reason

GUARDRAIL 3: Protocol Compliance
├─> Cross-references local triage protocols
└─> Citations link to source documents
```

### Audit Trail

Every action is logged:
- Session creation/modification
- Transcript generation
- CR generation requests
- RPA verification requests
- Approval/rejection decisions
- Execution results

---

## Technology Stack Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Next.js 15, React 19, Tailwind | User interface |
| Backend | FastAPI, Python 3.11+ | API & orchestration |
| Audio | Faster-Whisper (CTranslate2) | Speech transcription |
| LLM | Ollama + Parlant | Document generation |
| Vision | DeepSeek-VL2 | Screen OCR |
| RPA | PyAutoGUI | Keystroke injection |
| Vector DB | Qdrant | RAG protocol storage |
| Database | SQLite | Session & document storage |

---

## Deployment

**100% Local/Offline Operation**

All services run on localhost:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Ollama: `http://localhost:11434`
- Qdrant: `http://localhost:6333`

No cloud dependencies. No data leaves the clinic.
