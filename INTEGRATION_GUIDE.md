# Offline Triage MVP - Integration Guide

This guide explains how to run the integrated system with the React/Next.js frontend and FastAPI backend.

## Architecture

```
┌─────────────────────────────────────┐
│   Next.js Frontend (Port 3000)     │
│   - Landing Page                    │
│   - Patient Triage Flow             │
│   - Staff Portal                    │
└───────────────┬─────────────────────┘
                │ HTTP API Calls
                ↓
┌─────────────────────────────────────┐
│   FastAPI Backend (Port 8000)      │
│   - Triage Engine                   │
│   - Risk Assessment                 │
│   - Local LLM Processing            │
│   - SQLite Database                 │
└─────────────────────────────────────┘
```

## Prerequisites

1. **Python 3.8+** - For the backend
2. **Node.js 18+** - For the frontend
3. **LLM Model** (optional) - GGUF format model for AI summaries

## Quick Start

### 1. Set Up the Backend

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Download LLM model
mkdir -p models
cd models
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf -O llama-3.2-1b-instruct-q4_k_m.gguf
cd ..

# Start the backend
cd backend
python main.py
```

The backend will start on **http://localhost:8000**

### 2. Set Up the Frontend

Open a **new terminal** window:

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies (if not already done)
npm install

# Start the development server
npm run dev
```

The frontend will start on **http://localhost:3000**

### 3. Access the Application

- **Landing Page:** http://localhost:3000
- **Patient Triage:** http://localhost:3000/triage
- **Staff Portal:** http://localhost:3000/staff (Default PIN: 1234)

## Features

### Landing Page
- Modern, animated hero section
- Feature showcase
- FAQ section
- Links to triage and staff portal

### Patient Triage Flow
1. **Demographics Entry** - Age, sex, pregnancy status
2. **Chief Complaint Selection** - Choose primary concern
3. **Guided Questions** - Dynamic questionnaire based on complaint
4. **Risk Assessment** - Automatic risk band calculation (Red/Amber/Green)
5. **Summary & Ticket** - AI-generated summary with ticket ID

### Staff Portal
- **PIN Authentication** (default: 1234)
- **Case List** - View all triage cases
- **Filters** - By risk band, status, or search
- **Case Details** - Full patient information and assessment
- **AI Assistant** - Ask questions about specific cases
- **Status Updates** - Mark cases as reviewed or discharged

## Configuration

### Backend Configuration

Edit environment variables before starting:

```bash
# Backend configuration
export MODEL_PATH="models/llama-3.2-1b-instruct-q4_k_m.gguf"
export DB_PATH="data/triage.db"
export CONFIG_DIR="config"
export STAFF_PIN="1234"

# Then start backend
cd backend && python main.py
```

### Frontend Configuration

Edit `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Customizing Triage Rules

Edit `config/risk_rules.json` to modify risk assessment rules:

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

### Customizing Questions

Edit files in `config/triage_trees/` to modify questionnaires:

```json
{
  "id": "fever_check",
  "type": "yesno",
  "text": "Do you have a fever?",
  "next_if_yes": "temperature_question",
  "next_if_no": "other_symptoms"
}
```

## API Endpoints

### Patient Endpoints
- `POST /session/start` - Start new triage session
- `POST /session/{id}/demographics` - Submit demographics
- `POST /session/{id}/complaint` - Submit chief complaint
- `POST /session/{id}/answer` - Submit answer to question
- `GET /session/{id}/summary` - Get final summary

### Staff Endpoints
- `POST /staff/auth` - Authenticate with PIN
- `GET /staff/cases` - Get all cases (requires X-Staff-Pin header)
- `GET /staff/case/{id}` - Get case details
- `POST /staff/case/{id}/ask` - Ask AI question about case
- `POST /staff/case/{id}/status` - Update case status

## Building for Production

### Frontend

```bash
cd frontend
npm run build
npm run start  # Runs on port 3000
```

### Backend

The FastAPI backend can be deployed with:

```bash
# Using uvicorn directly
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Or with gunicorn for production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

## Troubleshooting

### Backend not connecting to frontend
- Check that backend is running on port 8000
- Verify CORS settings in `backend/main.py`
- Check `frontend/.env.local` has correct API URL

### LLM not loading
- Verify model file exists at specified path
- Check model is in GGUF format
- System will fallback to template-based responses if LLM fails

### Frontend build errors
- Run `npm install` in frontend directory
- Check Node.js version (requires 18+)
- Clear `.next` folder and rebuild

## Important Notes

⚠️ **DISCLAIMER:** This is a demonstration MVP, not validated for clinical use.

- Always have qualified medical staff review triage decisions
- Test thoroughly before any pilot deployment
- Consult with clinical experts when modifying rules
- Ensure HIPAA/GDPR compliance for your jurisdiction
- The AI cannot override deterministic risk rules

## Technology Stack

### Frontend
- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **shadcn/ui** - UI components

### Backend
- **FastAPI** - Python web framework
- **SQLite** - Local database
- **llama-cpp-python** - LLM inference
- **Pydantic** - Data validation

## License

This project is provided for educational and demonstration purposes.

**Not for clinical use without proper validation and regulatory approval.**

---

Built with ❤️ for healthcare innovation
