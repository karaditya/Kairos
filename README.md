# 🏥 Offline Triage Pre-Screen MVP

A fully functional, offline-first medical triage pre-screen system that runs entirely on your local machine. Features patient self-check-in, deterministic risk assessment, LLM-powered summaries, and staff helper mode.

**⚠️ DISCLAIMER: This tool does not provide medical diagnosis or treatment. A healthcare professional must review all cases.**

---

## 🎯 Features

### Patient Mode
- 📱 Touch-friendly interface for kiosks/tablets
- 🌍 Multi-language support (English, Spanish)
- 📋 Guided symptom questionnaires
- 🎫 Automatic ticket generation
- 🚦 Color-coded risk bands (Red/Amber/Green)

### Staff Mode
- 🔐 PIN-protected access
- 📊 Case list with filtering
- 💬 AI-powered Q&A about cases
- ✅ Status tracking (Pending/Reviewed/Discharged)
- 📝 Auto-generated clinical summaries

### Technical
- 🔒 100% offline operation (no internet required after setup)
- 🧠 Local LLM with TRM-style iterative reasoning
- 📜 Deterministic risk rules (model cannot override)
- 💾 SQLite database for data persistence
- 🔧 JSON-configurable triage trees

---

## 🚀 Quick Start

### 1. Clone/Download the Project

```bash
cd triage_mvp
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the LLM Model

You need a GGUF model file. Here are recommended options:

#### Option A: Llama 3.2 1B Instruct (Recommended - Smallest)

```bash
mkdir -p models
cd models

# Download from Hugging Face (choose Q4_K_M for best speed/quality balance)
wget https://huggingface.co/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Rename for consistency
mv Llama-3.2-1B-Instruct-Q4_K_M.gguf llama-3.2-1b-instruct-q4_k_m.gguf

cd ..
```

#### Option B: Phi-3.5 Mini Instruct (Better quality, larger)

```bash
mkdir -p models
cd models
wget https://huggingface.co/lmstudio-community/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
mv Phi-3.5-mini-instruct-Q4_K_M.gguf llama-3.2-1b-instruct-q4_k_m.gguf
cd ..
```

#### Option C: Using Hugging Face Hub CLI

```bash
pip install huggingface_hub
mkdir -p models
huggingface-cli download lmstudio-community/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --local-dir models
```

### 5. Run the Server

```bash
cd backend
python main.py
```

Or with custom settings:

```bash
MODEL_PATH="../models/llama-3.2-1b-instruct-q4_k_m.gguf" \
STAFF_PIN="9999" \
python main.py
```

### 6. Access the Interfaces

- **Patient Interface:** http://localhost:8000/
- **Staff Portal:** http://localhost:8000/staff (Default PIN: 1234)

---

## 📁 Project Structure

```
triage_mvp/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── database.py          # SQLite database layer
│   ├── triage_engine.py     # Question flow & risk rules
│   └── reasoning_engine.py  # TRM-style LLM reasoning
├── frontend/
│   ├── patient.html         # Patient check-in interface
│   └── staff.html           # Staff portal interface
├── config/
│   ├── risk_rules.json      # Deterministic risk rules
│   └── triage_trees/
│       ├── fever.json       # Fever/infection questions
│       └── chest_discomfort.json  # Chest pain questions
├── models/
│   └── (place GGUF model here)
├── data/
│   └── triage.db            # SQLite database (auto-created)
├── requirements.txt
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

1. Open http://localhost:8000/
2. Click "Begin Check-In"
3. Enter demographics (try age 1 for red flag)
4. Select "Fever / Infection"
5. Answer questions
6. View summary and ticket

### Test Staff Portal

1. Open http://localhost:8000/staff
2. Enter PIN: 1234
3. Select a case from the list
4. Ask: "Why was this patient flagged?"
5. Try marking case as reviewed

### Test Without Model

The system works without a model (fallback mode):

```bash
# Don't set MODEL_PATH, or point to non-existent file
python main.py
```

Fallback responses will be generated using templates.

---

## 🚧 Roadmap

- [ ] More chief complaint trees (abdominal pain, injury, etc.)
- [ ] PDF report generation
- [ ] Vital signs input (BP, pulse, SpO2)
- [ ] Multi-user staff authentication
- [ ] Integration with EMR systems
- [ ] Custom TRM model training

---

## 📄 License

This project is provided for educational and demonstration purposes. 

**Not for clinical use without proper validation and regulatory approval.**

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📞 Support

For questions about deployment or customization, please open an issue.

---

## ⚠️ Important Notes

1. **This is a demo/MVP** - not validated for clinical use
2. **Always have qualified medical staff** review triage decisions
3. **Test thoroughly** before any pilot deployment
4. **Consult with clinical experts** when modifying rules
5. **Ensure HIPAA/GDPR compliance** for your jurisdiction

---

Built with ❤️ for healthcare innovation
