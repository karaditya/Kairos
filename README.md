# Clinical Admin Edge (CAE) System

A 100% local/offline ambient administrative assistant for medical clinics. Automates clinical documentation by transcribing patient-clinician conversations, generating structured medical reports (Compte Rendu), and syncing data to EMR systems via RPA.

Built with **FastAPI** (backend) + **Next.js/React** (frontend) + **Ollama** (local LLM) + **Faster-Whisper** (speech-to-text).

**DISCLAIMER: This tool assists with administrative documentation only. It does not provide medical diagnosis or treatment. All clinical decisions must be made by qualified healthcare professionals.**

---

## Features

### Receptionist Portal
- Start/pause/resume/end patient sessions
- Real-time audio recording via WebSocket streaming
- Live transcript display with keyword highlighting
- Audio file upload for batch transcription
- Bilingual support (English/French)

### Clinician Portal
- Session list with status filtering (pending/reviewed/synced)
- Auto-generated Compte Rendu (structured clinical report)
- Editable sections with missing field alerts
- RAG-powered protocol citations
- Human-in-the-loop RPA sync approval
- PIN-protected access

### Admin Portal
- System health monitoring (6 backend services)
- Protocol management (ingest/search/clear)
- Ollama model management (list/select/pull)
- Smart GPU/CPU allocation display

### Technical Highlights
- **100% Offline** - No cloud dependencies after setup
- **Smart GPU/CPU Allocation** - Automatically optimizes for your hardware
- **Faster-Whisper** - GPU-accelerated speech recognition
- **Parlant Framework** - Agentic LLM for document generation
- **Qdrant Vector DB** - Embedded RAG for protocol retrieval
- **RPA Integration** - Automated EMR data entry with human approval

---

## Quick Start

### Prerequisites

- **Python 3.10+** for backend
- **Node.js 18+** for frontend
- **Ollama** for local LLM inference
- **NVIDIA GPU** (optional, for acceleration)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd triage_mvp
```

### 2. Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull mistral
ollama pull nomic-embed-text  # Required for embeddings
```

### 3. Start the Backend

```bash
chmod +x start-backend.sh
./start-backend.sh
```

The script will:
- Detect your GPU/CPU configuration
- Show smart allocation plan
- Create Python virtual environment
- Install dependencies
- Start Ollama if needed
- Launch the FastAPI server

### 4. Start the Frontend (in another terminal)

```bash
chmod +x start-frontend.sh
./start-frontend.sh
```

### 5. Access the Application

| URL | Description |
|-----|-------------|
| http://localhost:3000 | Home page (KAIROS landing) |
| http://localhost:3000/receptionist | Receptionist Portal |
| http://localhost:3000/clinician | Clinician Portal |
| http://localhost:3000/admin | Admin Portal |
| http://localhost:8000/docs | API Documentation |

**Default PIN:** 1234

---

## Smart GPU/CPU Allocation

The system automatically detects your hardware and allocates services optimally:

### High VRAM (8GB+)
| Service | Device | Configuration |
|---------|--------|---------------|
| Faster-Whisper | GPU | large-v3 model, float16 |
| Ollama LLM | GPU | All layers on GPU |
| Sentence Transformers | GPU | Embedding model |
| Qdrant/FastAPI/SQLite | CPU | I/O bound services |

### Medium VRAM (4-8GB)
| Service | Device | Configuration |
|---------|--------|---------------|
| Faster-Whisper | GPU | medium model, int8 |
| Ollama LLM | GPU+CPU | 20 layers on GPU |
| Sentence Transformers | CPU | Save GPU memory |
| Qdrant/FastAPI/SQLite | CPU | I/O bound services |

### Low VRAM (<4GB)
| Service | Device | Configuration |
|---------|--------|---------------|
| Faster-Whisper | GPU | small model, int8 |
| Ollama LLM | CPU | GPU memory limited |
| Sentence Transformers | CPU | GPU memory limited |
| Qdrant/FastAPI/SQLite | CPU | I/O bound services |

### CPU Only
| Service | Device | Configuration |
|---------|--------|---------------|
| All services | CPU | base/small models |

---

## Project Structure

```
triage_mvp/
├── backend/                    # FastAPI Backend (Python)
│   ├── main.py                 # Application entry point
│   ├── routers/                # API route handlers
│   │   ├── session.py          # Session management
│   │   ├── audio.py            # Audio/transcription
│   │   ├── vision.py           # Screen capture/OCR
│   │   ├── agent.py            # LLM agent/sync
│   │   └── admin.py            # System administration
│   ├── services/               # Business logic
│   │   ├── audio_service.py    # Faster-Whisper integration
│   │   ├── transcription.py    # Transcript processing
│   │   ├── parlant_agent.py    # Compte Rendu generation
│   │   ├── vision_service.py   # EMR screen reading
│   │   ├── rpa_service.py      # Robotic process automation
│   │   └── rag_service.py      # Protocol retrieval
│   ├── models/                 # Pydantic models
│   ├── db/                     # Database layer
│   └── requirements.txt        # Python dependencies
│
├── frontend/                   # Next.js Frontend (React)
│   ├── app/                    # Next.js 15 App Router
│   │   ├── page.tsx            # KAIROS landing page
│   │   ├── receptionist/       # Receptionist dashboard
│   │   ├── clinician/          # Clinician dashboard
│   │   └── admin/              # Admin dashboard
│   ├── components/ui/          # Reusable UI components
│   ├── hooks/                  # React hooks
│   │   └── useWebSocket.ts     # Audio streaming hook
│   └── lib/                    # Utilities
│       ├── api.ts              # CAE API client
│       ├── types.ts            # TypeScript interfaces
│       └── translations.ts     # i18n (EN/FR)
│
├── data/                       # Runtime data
│   ├── protocols/              # Protocol PDFs for RAG
│   └── cae.db                  # SQLite database
│
├── start-backend.sh            # Backend startup script
├── start-frontend.sh           # Frontend startup script
├── stop-backend.sh             # Shutdown script
└── requirements.txt            # Root dependencies
```

---

## API Endpoints

### Session Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/session/start` | Start new session |
| GET | `/session/` | List all sessions |
| GET | `/session/{id}` | Get session details |
| POST | `/session/{id}/pause` | Pause session |
| POST | `/session/{id}/resume` | Resume session |
| POST | `/session/{id}/end` | End session |

### Audio Processing
| Method | Endpoint | Description |
|--------|----------|-------------|
| WS | `/audio/stream/{id}` | WebSocket audio streaming |
| POST | `/audio/upload` | Upload audio file |
| GET | `/audio/transcript/{id}` | Get transcript |

### Vision (EMR Integration)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/vision/capture` | Capture screen region |
| POST | `/vision/calibrate` | Calibrate EMR coordinates |
| GET | `/vision/coordinates` | Get saved coordinates |

### Agent (Document Generation)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/agent/draft` | Generate Compte Rendu draft |
| POST | `/agent/sync` | Request RPA sync |
| POST | `/agent/sync/{id}/approve` | Approve sync request |
| POST | `/agent/sync/{id}/reject` | Reject sync request |
| GET | `/agent/pending` | List pending syncs |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/status` | System health check |
| POST | `/admin/protocols/ingest` | Ingest protocol PDF |
| POST | `/admin/protocols/search` | Search protocols |
| GET | `/admin/protocols/stats` | Protocol statistics |
| DELETE | `/admin/protocols/clear` | Clear protocol index |
| GET | `/admin/models` | List Ollama models |
| POST | `/admin/models/select` | Select active model |
| GET | `/admin/models/pull/{name}` | Pull new model (SSE) |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `mistral` | Default LLM model |
| `WHISPER_MODEL_SIZE` | Auto | Whisper model (base/small/medium/large-v3) |
| `WHISPER_DEVICE` | Auto | Device (cuda/cpu) |
| `WHISPER_COMPUTE_TYPE` | Auto | Compute type (float16/int8) |
| `EMBEDDING_DEVICE` | Auto | Embedding device (cuda/cpu) |
| `CAE_PIN` | `1234` | Portal access PIN |
| `DB_PATH` | `data/cae.db` | SQLite database path |

---

## Workflow

### 1. Receptionist Flow
```
Start Session → Record Audio → Live Transcript → End Session
```

### 2. Clinician Flow
```
Select Session → Generate Draft → Edit Compte Rendu → Request Sync → Approve
```

### 3. RPA Sync (with Human-in-the-Loop)
```
Draft Generated → Preview Sync → Human Approval → RPA Executes → Verify
```

---

## Technology Stack

### Backend
- **FastAPI** - Async web framework
- **Faster-Whisper** - CTranslate2-based speech recognition
- **Parlant** - Agentic LLM framework
- **Ollama** - Local LLM inference
- **Qdrant** - Embedded vector database
- **PyAutoGUI** - Cross-platform RPA

### Frontend
- **Next.js 15** - React framework
- **React 19** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Radix UI** - Accessible components

### Infrastructure
- **100% Local** - No cloud dependencies
- **SQLite** - Embedded database
- **WebSocket** - Real-time audio streaming
- **CUDA** - Optional GPU acceleration

---

## Troubleshooting

### Backend won't start
```bash
# Check Python version (requires 3.10+)
python3 --version

# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Ollama not responding
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
```

### GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA toolkit (Ubuntu)
sudo apt install nvidia-cuda-toolkit
```

### Audio not working
```bash
# Check microphone permissions
# Browser must allow microphone access for WebSocket streaming

# For file upload, check supported formats:
# WAV, MP3, M4A, FLAC, OGG
```

### Frontend can't connect
```bash
# Verify backend is running
curl http://localhost:8000/admin/status

# Check CORS (backend allows localhost:3000)
```

---

## Stopping the System

```bash
./stop-backend.sh
```

This will:
- Stop FastAPI server
- Free port 8000
- Show GPU/memory status
- Note: Ollama is not stopped (may be used by other apps)

To stop Ollama manually:
```bash
pkill ollama
```

---

## Security Notes

1. **Local Only** - All data stays on the machine
2. **PIN Protected** - Clinician/Admin portals require PIN
3. **Human-in-the-Loop** - RPA sync requires explicit approval
4. **No PHI Transmission** - Audio processed locally
5. **Audit Trail** - All actions logged

---

## License

This project is provided for educational and demonstration purposes.

**Not for clinical use without proper validation and regulatory approval.**

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

Built with care for healthcare innovation
