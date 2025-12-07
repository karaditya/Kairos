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
- 💬 **AI-powered Q&A with reasoning visualization**
  - 🟣 **Purple box**: Chain-of-thought reasoning (collapsible)
  - 🔵 **Blue box**: Final answer
  - 🟢 **Green box**: Suggested follow-up questions
- 🔄 **Dynamic model switching** - Switch between 10+ models on the fly
- ✅ Status tracking (Pending/Reviewed/Discharged)
- 📝 Auto-generated clinical summaries
- 🎨 Modern UI with Tailwind CSS and animations

### Technical
- 🔒 100% offline operation (no internet required after setup)
- 🧠 **Multi-model support** with 10+ offline LLMs
- 🔄 **Dynamic model switching** during runtime (no restart needed)
- 🧬 **Chain-of-thought reasoning** - See how AI thinks (DeepSeek models)
- 📜 Deterministic risk rules (model cannot override)
- 🔐 **XSS protection** with HTML sanitization
- 💾 SQLite database for data persistence
- 🔧 JSON-configurable triage trees
- ⚡ Fast API backend with async operations
- 🎯 TypeScript for type safety

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** for backend
- **Node.js 18+** and **npm** for frontend
- **NVIDIA CUDA Toolkit** (optional, for GPU acceleration) - Ubuntu: `sudo apt install nvidia-cuda-toolkit`

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

# Install dependencies (CPU-only by default)
pip install -r requirements.txt
```

**Optional: Enable GPU Acceleration**

If you have an NVIDIA GPU with CUDA support, you can enable GPU acceleration for faster inference:

**Step 1: Install CUDA Toolkit** (one-time setup)
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-cuda-toolkit -y

# Verify installation
nvcc --version
```

**Step 2: Build llama-cpp-python with CUDA**
```bash
source venv/bin/activate
pip uninstall llama-cpp-python -y
CUDACXX=/usr/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall
```

**Step 3: Verify GPU support**
```bash
# Check for CUDA library
ls venv/lib/python3.12/site-packages/llama_cpp/lib/libggml-cuda.so
```

This builds llama-cpp-python with CUDA support (~3-5 minutes). You can still use CPU mode by setting `N_GPU_LAYERS=0`.

#### Download LLM Models (Optional but Recommended)

The system supports **10+ offline models** with dynamic model switching. You can download one or multiple models.

**Supported Models:**

| Model | Family | Size | Speed | Quality | Best For |
|-------|--------|------|-------|---------|----------|
| **Llama 3.2 1B** ⭐ | Meta | 738MB | ⚡⚡⚡ | ⭐⭐ | CPU-only, fast inference |
| **Llama 3.2 3B** | Meta | 2GB | ⚡⚡ | ⭐⭐⭐ | Balanced quality/speed |
| **Gemma 2 2B** | Google | 1.6GB | ⚡⚡⚡ | ⭐⭐⭐ | Google's compact model |
| **DeepSeek R1 1.5B** 🧠 | DeepSeek | 1GB | ⚡⚡⚡ | ⭐⭐⭐ | Chain-of-thought reasoning |
| **DeepSeek R1 7B** | DeepSeek | 4.4GB | ⚡ | ⭐⭐⭐⭐ | Advanced reasoning (GPU) |
| **Phi 3.5 Mini** | Microsoft | 2.5GB | ⚡⚡ | ⭐⭐⭐⭐ | High quality, moderate size |
| **Qwen 2.5 1.5B** | Alibaba | 1GB | ⚡⚡⚡ | ⭐⭐⭐ | Efficient, good reasoning |
| **Qwen 2.5 3B** | Alibaba | 2GB | ⚡⚡ | ⭐⭐⭐⭐ | Excellent medical reasoning |
| **SmolLM2 1.7B** | HuggingFace | 1GB | ⚡⚡⚡ | ⭐⭐ | Lightweight, efficient |
| **MedLlama 3** 🏥 | Medical | 4.7GB | ⚡ | ⭐⭐⭐⭐⭐ | Medical-specific (GPU) |

⭐ = Recommended for beginners | 🧠 = Shows reasoning process | 🏥 = Medical-specialized

**Quick Download (Recommended for CPU):**

```bash
# Create models directory
mkdir -p models
cd models

# Download Llama 3.2 1B (fastest, smallest)
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -O llama-3.2-1b-instruct-q4_k_m.gguf

cd ..
```

**Download Multiple Models:**

```bash
mkdir -p models
cd models

# Llama 3.2 1B (CPU-friendly)
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -O llama-3.2-1b-instruct-q4_k_m.gguf

# DeepSeek R1 1.5B (shows chain-of-thought reasoning)
wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
  -O deepseek-r1-1.5b-q4_k_m.gguf

# Qwen 2.5 1.5B (good balance)
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

cd ..
```

**Download with Hugging Face CLI (Alternative):**

```bash
pip install huggingface_hub
mkdir -p models

# Example: Download Llama 3.2 1B
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --local-dir models \
    --local-dir-use-symlinks False

# Rename to expected filename
mv models/Llama-3.2-1B-Instruct-Q4_K_M.gguf models/llama-3.2-1b-instruct-q4_k_m.gguf
```

**Model Switching:**

Once you've downloaded multiple models, you can switch between them in the Staff Portal:
1. Open http://localhost:3000/staff
2. Look for the model selector dropdown (CPU icon) in the top-right
3. Select any downloaded model
4. The system will automatically switch and use the new model

**Note:** The system works in fallback mode without models, but AI-powered features won't be available.

**Expected Model Filenames:**

The system looks for models with these exact filenames in the `models/` directory:

| Model | Expected Filename |
|-------|-------------------|
| Llama 3.2 1B | `llama-3.2-1b-instruct-q4_k_m.gguf` |
| Llama 3.2 3B | `llama-3.2-3b-instruct-q4_k_m.gguf` |
| Gemma 2 2B | `gemma-2-2b-instruct-q4_k_m.gguf` |
| DeepSeek R1 1.5B | `deepseek-r1-1.5b-q4_k_m.gguf` |
| DeepSeek R1 7B | `deepseek-r1-7b-q4_k_m.gguf` |
| Phi 3.5 Mini | `phi-3.5-mini-instruct-q4_k_m.gguf` |
| Qwen 2.5 1.5B | `qwen2.5-1.5b-instruct-q4_k_m.gguf` |
| Qwen 2.5 3B | `qwen2.5-3b-instruct-q4_k_m.gguf` |
| SmolLM2 1.7B | `smollm2-1.7b-instruct-q4_k_m.gguf` |
| MedLlama3 v20 | `medllama3-v20-q4_k_m.gguf` |

After downloading, rename files to match these exact names. The system automatically detects which models are available.

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

**Terminal 1 - Backend (CPU mode - default):**
```bash
./start-backend.sh
```

**Terminal 1 - Backend (GPU mode - if CUDA-enabled):**
```bash
N_GPU_LAYERS=-1 ./start-backend.sh
```

**Terminal 2 - Frontend:**
```bash
./start-frontend.sh
```

#### Option B: Manual Start

**Terminal 1 - Backend (CPU mode):**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd backend
python main.py
```

**Terminal 1 - Backend (GPU mode):**
```bash
source venv/bin/activate
cd backend
N_GPU_LAYERS=-1 python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

#### Environment Variable Options

You can customize the backend behavior using environment variables:

```bash
# GPU Acceleration (requires CUDA-enabled llama-cpp-python)
N_GPU_LAYERS=-1 python backend/main.py       # All layers on GPU
N_GPU_LAYERS=20 python backend/main.py       # 20 layers on GPU, rest on CPU
N_GPU_LAYERS=0 python backend/main.py        # CPU only (default)

# Change staff PIN
STAFF_PIN="9999" python backend/main.py

# Use custom model path
MODEL_PATH="../models/my-model.gguf" python backend/main.py

# Combine multiple settings
N_GPU_LAYERS=-1 STAFF_PIN="9999" python backend/main.py
```

### 5. Access the Application

- **Patient Interface:** http://localhost:3000/triage
- **Staff Portal:** http://localhost:3000/staff (Default PIN: 1234)
- **Home Page:** http://localhost:3000/
- **Backend API:** http://localhost:8000 (API docs at http://localhost:8000/docs)

---

## ✨ What's New

### Recent Features (Latest Update)

#### 🧬 Chain-of-Thought Reasoning Visualization
- **Purple Box**: See how the AI thinks through problems (DeepSeek models)
- **Collapsible**: Click to expand/hide the reasoning process
- Helps staff understand AI decision-making

#### 🔄 Dynamic Model Switching
- Switch between 10+ models without restarting
- Compare responses from different models
- Model selector in staff portal (top-right dropdown)

#### 🎨 Improved Answer Display
- **Blue Box**: Main answer
- **Green Box**: Suggested follow-up questions (automatically extracted)
- **Purple Box**: Chain-of-thought reasoning
- Clean, organized UI with color-coded sections

#### 🔐 Security Enhancements
- **XSS Protection**: All LLM outputs are HTML-escaped
- **Logging**: Parsing failures now logged for debugging
- **Multi-section Support**: Handles complex model outputs gracefully

#### 📊 Model Management API
- `/models` endpoint lists all supported models
- `/models/switch` allows runtime model switching
- `/models/stats` shows engine statistics
- Check availability before downloading

---

## 📁 Project Structure

```
triage_mvp/
├── backend/                      # FastAPI Backend
│   ├── main.py                   # FastAPI application & API routes
│   ├── database.py               # SQLite database layer
│   ├── triage_engine.py          # Question flow & risk rules
│   ├── multi_model_engine.py     # Multi-model LLM reasoning engine
│   └── model_registry.py         # Supported models configuration
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
| `N_GPU_LAYERS` | `0` | GPU layers: `0` = CPU only, `-1` = all GPU, or specify number |

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
- **GGUF Models** - Quantized models for efficient CPU/GPU inference
- **CUDA Support** - Optional GPU acceleration with NVIDIA GPUs

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
| POST | `/staff/case/{id}/ask` | Ask question (with optional `model_id`) |
| POST | `/staff/case/{id}/status` | Update case status |

### Model Management Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/models` | List all supported models |
| GET | `/models/current` | Get currently loaded model |
| GET | `/models/{model_id}` | Get specific model info |
| POST | `/models/switch` | Switch to different model (staff only) |
| GET | `/models/stats` | Get engine statistics (staff only) |

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
5. **Test AI Q&A:**
   - Ask: "Why was this patient flagged?"
   - Look for **3 colored boxes**:
     - 🟣 Purple = Chain-of-thought reasoning (click to expand)
     - 🔵 Blue = Final answer
     - 🟢 Green = Suggested follow-up questions
6. **Test model switching:**
   - Click the model dropdown (CPU icon, top-right)
   - Switch to a different model (if you downloaded multiple)
   - Ask another question to see the new model's response
7. Try marking case as reviewed/discharged
8. Toggle dark mode to test theme switching

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
- **Verify model files exist:**
  ```bash
  ls -lh models/*.gguf
  ```
- **Check model registry:** Visit http://localhost:8000/models to see available models
- **Download models:** Use wget commands from the "Download LLM Models" section
- **Check file permissions:** Ensure model files are readable
- **Fallback mode:** System works without models - you'll see a warning message
- **Model switching:** Staff portal shows which models are available vs. loaded

### GPU acceleration not working
- **Install CUDA toolkit first:**
  ```bash
  sudo apt install nvidia-cuda-toolkit -y
  nvcc --version  # Should show CUDA version
  ```
- **Rebuild llama-cpp-python with CUDA:**
  ```bash
  source venv/bin/activate
  pip uninstall llama-cpp-python -y
  CUDACXX=/usr/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall
  ```
- **Verify CUDA library exists:**
  ```bash
  ls venv/lib/python3.12/site-packages/llama_cpp/lib/libggml-cuda.so
  ```
- **Check GPU memory usage:** `nvidia-smi` should show ~1GB used when model loads with `N_GPU_LAYERS=-1`

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
