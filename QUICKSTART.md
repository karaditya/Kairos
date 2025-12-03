# Quick Start Guide - Integrated Triage System

## Overview

Your triage system now has a beautiful React/Next.js frontend integrated with the Python FastAPI backend!

## How to Run

### Option 1: Using the Startup Scripts (Recommended)

**Terminal 1 - Backend:**
```bash
./start-backend.sh
```

**Terminal 2 - Frontend:**
```bash
./start-frontend.sh
```

### Option 2: Manual Start

**Backend (Terminal 1):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd backend && python main.py
```

**Frontend (Terminal 2):**
```bash
cd frontend
npm install
npm run dev
```

## Access the Application

Once both servers are running:

- **Landing Page:** http://localhost:3000
- **Patient Triage:** http://localhost:3000/triage
- **Staff Portal:** http://localhost:3000/staff (PIN: 1234)
- **Backend API:** http://localhost:8000 (for API docs: http://localhost:8000/docs)

## Application Flow

### 1. Landing Page (http://localhost:3000)
- Modern, animated hero section
- Feature showcase
- Click "Start Triage" to begin patient assessment
- Click "Staff Portal" to access staff interface

### 2. Patient Triage Flow (http://localhost:3000/triage)

**Step 1: Demographics**
- Enter age, sex, pregnancy status
- Touch-friendly interface

**Step 2: Chief Complaint**
- Select primary concern (Fever/Infection, Chest Pain, etc.)
- Add additional details

**Step 3: Guided Questions**
- Dynamic questionnaire based on complaint
- Yes/No, multiple choice, and numeric questions
- Progress bar shows completion

**Step 4: Summary**
- Risk band (Red/Amber/Green) with color coding
- AI-generated clinical summary
- Key flags and warnings
- Ticket ID for staff reference
- Printable summary

### 3. Staff Portal (http://localhost:3000/staff)

**Login:**
- Default PIN: 1234
- Can be changed via STAFF_PIN environment variable

**Features:**
- View all triage cases
- Filter by risk band (Red/Amber/Green) or status
- Search by ticket ID
- View detailed case information
- Ask AI questions about specific cases
- Update case status (Pending → Reviewed → Discharged)

## Key Features

### 🔒 100% Offline Operation
- All processing on local machine
- No data leaves your device
- Optional LLM for better summaries

### 🚦 Risk Assessment
- Deterministic rules (AI cannot override)
- Red/Amber/Green color coding
- Configurable via JSON

### 🧠 Local AI Processing
- TRM-style iterative reasoning
- GGUF format models
- Fallback mode without LLM

### 📱 Modern UI
- Responsive design
- Smooth animations (Framer Motion)
- Dark mode support
- Touch-friendly for tablets/kiosks

## Testing the System

### Test Patient Flow:
1. Go to http://localhost:3000
2. Click "Start Triage"
3. Enter: Age 1, Sex Male → Should trigger red flag
4. Complete the questionnaire
5. View summary and ticket

### Test Staff Portal:
1. Go to http://localhost:3000/staff
2. Enter PIN: 1234
3. Select a case from the list
4. Ask: "Why was this patient flagged?"
5. Update case status to "Reviewed"

## Configuration

### Change Staff PIN:
```bash
export STAFF_PIN="9999"
./start-backend.sh
```

### Use Different Model:
```bash
export MODEL_PATH="path/to/your/model.gguf"
./start-backend.sh
```

### Change Frontend API URL:
Edit `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://your-backend-url:8000
```

## Customization

### Modify Risk Rules
Edit: `config/risk_rules.json`

### Modify Questions
Edit: `config/triage_trees/*.json`

### Change Styling
Edit: `frontend/app/globals.css` and components

## Troubleshooting

**Backend not starting?**
- Check Python 3.8+ installed: `python3 --version`
- Install dependencies: `pip install -r requirements.txt`

**Frontend not building?**
- Check Node.js 18+ installed: `node --version`
- Clear cache: `rm -rf frontend/.next frontend/node_modules`
- Reinstall: `cd frontend && npm install`

**Can't connect to backend from frontend?**
- Check backend is running on port 8000
- Verify `frontend/.env.local` has correct API URL
- Check browser console for CORS errors

**LLM not loading?**
- System will fallback to templates (still works!)
- Download model: See INTEGRATION_GUIDE.md
- Check model path is correct

## Production Deployment

### Build Frontend:
```bash
cd frontend
npm run build
npm run start  # Production server on port 3000
```

### Deploy Backend:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Important Notes

⚠️ **MEDICAL DISCLAIMER**

This is a demonstration/MVP tool:
- NOT validated for clinical use
- NOT a medical device
- NOT a substitute for professional medical advice
- Requires qualified healthcare professional review

Always test thoroughly and consult clinical experts before any pilot deployment.

## Architecture

```
┌───────────────────────────┐
│   Next.js Frontend        │
│   Port 3000               │
│   - React components      │
│   - Tailwind CSS          │
│   - Framer Motion         │
└─────────┬─────────────────┘
          │ HTTP REST API
          ↓
┌───────────────────────────┐
│   FastAPI Backend         │
│   Port 8000               │
│   - Triage Engine         │
│   - Risk Assessment       │
│   - Local LLM             │
│   - SQLite Database       │
└───────────────────────────┘
```

## Tech Stack

- **Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS
- **Backend:** FastAPI, Python 3.8+, llama-cpp-python
- **Database:** SQLite
- **AI:** GGUF format models (Llama 3.2 recommended)

---

🎉 **You're ready to go!**

Run `./start-backend.sh` and `./start-frontend.sh` in separate terminals, then visit http://localhost:3000

For detailed information, see `INTEGRATION_GUIDE.md`
