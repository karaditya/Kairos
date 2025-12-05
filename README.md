# 🏥 Offline Triage Pre-Screen MVP

A fully functional, offline-first medical triage pre-screen system that runs entirely on your local machine. Features patient self-check-in, deterministic risk assessment, LLM-powered summaries, and staff helper mode.

Built with **FastAPI** (backend) + **Next.js/React** (frontend) + **Local LLM**.

**⚠️ DISCLAIMER: This tool does not provide medical diagnosis or treatment. A healthcare professional must review all cases.**

---

## 🎯 Features

### Patient Mode
- 📱 Modern, responsive React interface optimized for kiosks/tablets
- 🌍 Multi-language support (English, Spanish)
- 📋 Guided symptom questionnaires with smooth animations
- 🎫 Automatic ticket generation
- 🚦 Color-coded risk bands (Red/Amber/Green)
- 🌓 Dark mode support

### Staff Mode
- 🔐 PIN-protected access
- 📊 Real-time case list with filtering and search
- 💬 AI-powered Q&A about cases
- ✅ Status tracking (Pending/Reviewed/Discharged)
- 📝 Auto-generated clinical summaries
- 🎨 Modern UI with Tailwind CSS

### Technical
- 🔒 100% offline operation (no internet required after setup)
- 🧠 Local LLM with TRM-style iterative reasoning
- 📜 Deterministic risk rules (model cannot override)
- 💾 SQLite database for data persistence
- 🔧 JSON-configurable triage trees
- ⚡ Fast API backend with async operations
- 🎯 TypeScript for type safety

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** for backend
- **Node.js 18+** and **npm** for frontend

### 1. Clone/Download the Project

```bash
cd triage_mvp
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Download the LLM Model (Optional but Recommended)

You need a GGUF model file. Choose one of these options:

**Option A: Llama 3.2 1B Instruct (Recommended - Smallest, ~738MB)**

```bash
mkdir -p models
cd models

# Download from Hugging Face (bartowski's reliable repository)
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf -O llama-3.2-1b-instruct-q4_k_m.gguf

cd ..
```

**Option B: Phi-3.5 Mini Instruct (Better quality, ~2.5GB)**

```bash
mkdir -p models
cd models
wget https://huggingface.co/lmstudio-community/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
mv Phi-3.5-mini-instruct-Q4_K_M.gguf llama-3.2-1b-instruct-q4_k_m.gguf
cd ..
```

**Option C: Using Hugging Face Hub CLI**

```bash
pip install huggingface_hub
mkdir -p models
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --local-dir models \
    --local-dir-use-symlinks False
mv models/Llama-3.2-1B-Instruct-Q4_K_M.gguf models/llama-3.2-1b-instruct-q4_k_m.gguf
```

**Note:** The system works in fallback mode without a model, but LLM-powered summaries won't be available.

### 3. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

cd ..
```

### 4. Start the Application

#### Option A: Using Startup Scripts (Recommended)

Open two terminal windows:

**Terminal 1 - Backend:**
```bash
./start-backend.sh
```

**Terminal 2 - Frontend:**
```bash
./start-frontend.sh
```

#### Option B: Manual Start

**Terminal 1 - Backend:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

#### Custom Backend Settings

```bash
MODEL_PATH="../models/llama-3.2-1b-instruct-q4_k_m.gguf" \
STAFF_PIN="9999" \
python backend/main.py
```

### 5. Access the Application

- **Patient Interface:** http://localhost:3000/triage
- **Staff Portal:** http://localhost:3000/staff (Default PIN: 1234)
- **Home Page:** http://localhost:3000/
- **Backend API:** http://localhost:8000 (API docs at http://localhost:8000/docs)

---

## 📁 Project Structure

```
triage_mvp/
├── backend/                      # FastAPI Backend
│   ├── main.py                   # FastAPI application & API routes
│   ├── database.py               # SQLite database layer
│   ├── triage_engine.py          # Question flow & risk rules
│   └── reasoning_engine.py       # TRM-style LLM reasoning
│
├── frontend/                     # Next.js Frontend
│   ├── app/                      # Next.js 13+ app directory
│   │   ├── layout.tsx            # Root layout with theme provider
│   │   ├── page.tsx              # Home page (landing)
│   │   ├── triage/
│   │   │   └── page.tsx          # Patient triage flow
│   │   └── staff/
│   │       └── page.tsx          # Staff portal
│   ├── components/
│   │   └── ui/                   # Reusable UI components
│   │       ├── button.tsx        # Button component
│   │       ├── card.tsx          # Card component
│   │       ├── input.tsx         # Input component
│   │       ├── accordion.tsx     # Accordion component
│   │       └── theme-toggle.tsx  # Dark mode toggle
│   ├── lib/
│   │   ├── api.ts                # API client for backend
│   │   └── utils.ts              # Utility functions
│   ├── package.json              # Node.js dependencies
│   ├── tsconfig.json             # TypeScript config
│   ├── tailwind.config.ts        # Tailwind CSS config
│   └── next.config.js            # Next.js config
│
├── config/                       # Configuration Files
│   ├── risk_rules.json           # Deterministic risk rules
│   └── triage_trees/
│       ├── fever.json            # Fever/infection questions
│       └── chest_discomfort.json # Chest pain questions
│
├── models/                       # LLM Models
│   └── llama-3.2-1b-instruct-q4_k_m.gguf  (download separately)
│
├── data/                         # Application Data
│   └── triage.db                 # SQLite database (auto-created)
│
├── venv/                         # Python virtual environment
├── requirements.txt              # Python dependencies
├── start-backend.sh              # Backend startup script
├── start-frontend.sh             # Frontend startup script
└── README.md
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/llama-3.2-1b-instruct-q4_k_m.gguf` | Path to GGUF model |
| `DB_PATH` | `data/triage.db` | SQLite database path |
| `CONFIG_DIR` | `config` | Directory for JSON configs |
| `STAFF_PIN` | `1234` | Staff portal PIN |

### Risk Rules

Edit `config/risk_rules.json` to modify triage rules:

```json
{
  "id": "high_fever",
  "description": "Temperature above 40°C",
  "band": "red",
  "priority": 90,
  "conditions": [
    {"field": "answers.temperature_value", "op": ">=", "value": 40}
  ]
}
```

**Operators:** `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `contains`, `is_true`, `is_false`

### Triage Trees

Edit files in `config/triage_trees/` to modify questions:

```json
{
  "id": "fever_check",
  "type": "yesno",
  "text": "Do you have a fever?",
  "next_if_yes": "temperature_question",
  "next_if_no": "other_symptoms"
}
```

**Question types:** `yesno`, `choice`, `numeric`, `text`

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **Uvicorn** - Lightning-fast ASGI server
- **Pydantic** - Data validation using Python type annotations
- **llama-cpp-python** - Python bindings for llama.cpp (local LLM inference)
- **SQLite** - Lightweight, serverless database

### Frontend
- **Next.js 15** - React framework with App Router
- **React 19** - UI library
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **Radix UI** - Accessible component primitives
- **Lucide React** - Beautiful icon library

### Infrastructure
- **100% Local** - No cloud dependencies
- **Offline-First** - Works without internet
- **SQLite Database** - No database server needed
- **GGUF Models** - Quantized models for efficient CPU inference

---

## 🧠 TRM-Style Reasoning

The system implements iterative reasoning similar to Transformer Reasoning Machines:

1. **Initialize** latent state `z` and output state `y`
2. **Loop** T times (default 3):
   - Update reasoning: `z = reason(z, y, patient_data)`
3. **Generate** final summary from refined reasoning

This produces more thoughtful, coherent summaries than single-shot generation.

**Important:** The LLM can only summarize data and suggest questions. Risk bands are computed by deterministic rules that the model cannot override.

---

## 🔌 API Endpoints

### Patient Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/session/start` | Start new session |
| POST | `/session/{id}/demographics` | Submit demographics |
| POST | `/session/{id}/complaint` | Select chief complaint |
| POST | `/session/{id}/answer` | Submit question answer |
| GET | `/session/{id}/summary` | Get final summary |

### Staff Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/staff/auth` | Authenticate with PIN |
| GET | `/staff/cases` | List all cases |
| GET | `/staff/case/{id}` | Get case details |
| POST | `/staff/case/{id}/ask` | Ask question about case |
| POST | `/staff/case/{id}/status` | Update case status |

---

## 🔒 Safety Features

1. **No Diagnosis:** System cannot diagnose or recommend treatment
2. **Deterministic Rules:** Risk bands computed by rules, not AI
3. **Mandatory Disclaimer:** Every output includes medical disclaimer
4. **Audit Trail:** All answers stored for review
5. **Offline Operation:** Patient data never leaves the device

---

## 🧪 Testing

### Test Patient Flow

1. Open http://localhost:3000/triage
2. Click "Begin Check-In"
3. Enter demographics (try age 1 for red flag)
4. Select "Fever / Infection"
5. Answer questions with smooth animated transitions
6. View summary and ticket with color-coded risk band

### Test Staff Portal

1. Open http://localhost:3000/staff
2. Enter PIN: 1234 (default)
3. Browse the case list with real-time updates
4. Click on a case to view details
5. Ask: "Why was this patient flagged?"
6. Try marking case as reviewed/discharged
7. Toggle dark mode to test theme switching

### Test Without Model (Fallback Mode)

The system works without an LLM model:

```bash
# Backend will run in fallback mode if model file doesn't exist
./start-backend.sh
```

In fallback mode:
- Triage flow works normally
- Risk assessment is fully functional (rule-based)
- LLM-powered summaries are replaced with template-based responses
- Staff Q&A provides basic template answers

### API Testing

Access interactive API documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🚧 Roadmap

### Features
- [ ] More chief complaint trees (abdominal pain, injury, respiratory, etc.)
- [ ] PDF report generation for patient summaries
- [ ] Vital signs input (BP, pulse, SpO2, temperature)
- [ ] Multi-user staff authentication and role-based access
- [ ] Real-time WebSocket updates for staff dashboard
- [ ] Print-friendly ticket format for kiosk printers

### Technical Improvements
- [ ] Integration with EMR/EHR systems (FHIR support)
- [ ] Custom TRM model training pipeline
- [ ] Progressive Web App (PWA) for offline mobile support
- [ ] Internationalization (i18n) for additional languages
- [ ] Docker containerization for easier deployment
- [ ] Unit and integration test coverage
- [ ] Analytics dashboard for triage metrics

---

## 📄 License

This project is provided for educational and demonstration purposes. 

**Not for clinical use without proper validation and regulatory approval.**

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

### Development Workflow

1. **Fork the repository** and clone your fork
2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Make your changes**
   - Backend: Edit Python files in `backend/`
   - Frontend: Edit React/TypeScript files in `frontend/`
   - Configuration: Modify JSON files in `config/`
4. **Test your changes**
   - Start both backend and frontend
   - Test the affected functionality
5. **Commit and push**
   ```bash
   git add .
   git commit -m "Add my new feature"
   git push origin feature/my-new-feature
   ```
6. **Submit a pull request**

### Code Style
- **Python:** Follow PEP 8 conventions
- **TypeScript/React:** Use TypeScript types, follow React best practices
- **Comments:** Add docstrings for functions, especially in backend logic
- **Testing:** Add tests for new features when applicable

---

## 📞 Support

For questions about deployment or customization, please open an issue.

---

## 🔧 Troubleshooting

### Backend won't start
- **Check Python version:** Requires Python 3.10+
  ```bash
  python3 --version
  ```
- **Activate virtual environment:** Make sure venv is activated
  ```bash
  source venv/bin/activate
  ```
- **Install dependencies:** Re-run pip install
  ```bash
  pip install -r requirements.txt
  ```

### Frontend won't start
- **Check Node.js version:** Requires Node.js 18+
  ```bash
  node --version
  ```
- **Clear dependencies and reinstall:**
  ```bash
  cd frontend
  rm -rf node_modules package-lock.json
  npm install
  ```

### Frontend can't connect to backend
- **Check CORS settings:** Backend should allow requests from `http://localhost:3000`
- **Verify backend is running:** Visit http://localhost:8000/docs
- **Check API URL:** Frontend uses `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`)

### Model not loading
- **Verify model path:** Check that the model file exists at `models/llama-3.2-1b-instruct-q4_k_m.gguf`
- **Check file permissions:** Ensure model file is readable
- **Fallback mode:** System works without model - you'll see a warning message

### Database issues
- **Reset database:** Delete `data/triage.db` and restart backend (auto-recreates)
- **Check permissions:** Ensure `data/` directory is writable

---

## ⚠️ Important Notes

1. **This is a demo/MVP** - not validated for clinical use
2. **Always have qualified medical staff** review triage decisions
3. **Test thoroughly** before any pilot deployment
4. **Consult with clinical experts** when modifying rules
5. **Ensure HIPAA/GDPR compliance** for your jurisdiction

---

Built with ❤️ for healthcare innovation
