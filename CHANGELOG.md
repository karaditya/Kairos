# Changelog - Triage MVP

## Latest Updates (December 2025)

### 🔐 Security & Robustness Improvements

#### XSS Protection (CRITICAL)
- **Added HTML sanitization** to all LLM outputs using `html.escape()`
- Prevents script injection attacks from malicious model outputs
- All reasoning and answer text now properly escaped

#### Logging & Debugging
- **Added comprehensive logging** for parsing failures
- Logs when REASONING/ANSWER markers are in wrong order
- Logs when multiple sections detected or partial markers found
- Makes debugging easier with clear warning messages

#### Multi-Section Handling
- Improved parsing to handle models that output multiple REASONING/ANSWER pairs
- Uses first valid pair and logs warning about extras
- Prevents content mixing and confusion

**Files Modified:**
- `backend/multi_model_engine.py` - Added HTML escaping, logging, multi-section detection

---

### 🧬 Chain-of-Thought Reasoning Visualization

#### Frontend Display
- **Purple Box**: Shows AI's thinking process (collapsible)
- **Blue Box**: Shows final answer
- **Green Box**: Shows suggested follow-up questions
- Staff can now see how the AI reached its conclusion

#### Backend Parsing
- Enhanced `parse_reasoning_response()` to detect multiple formats:
  - `<think>...</think>` tags (DeepSeek models)
  - `REASONING:` / `ANSWER:` section headers
  - `ANALYSIS:`, `THINKING:`, `THOUGHT PROCESS:` alternatives
- Extracts suggested questions from `FOLLOW-UP QUESTIONS:` section

**Files Modified:**
- `backend/multi_model_engine.py` - Enhanced parsing logic
- `frontend/app/staff/page.tsx` - 3-box display system

---

### 🔄 Multi-Model Support & Dynamic Switching

#### 10+ Offline Models Supported
- **Llama** 3.2 (1B, 3B)
- **Gemma** 2 (2B)
- **DeepSeek** R1 (1.5B, 7B) - with chain-of-thought
- **Phi** 3.5 Mini
- **Qwen** 2.5 (1.5B, 3B)
- **SmolLM** 2 (1.7B)
- **MedLlama** 3 v20 - medical-specialized

#### Runtime Model Switching
- Switch between models without restarting server
- Staff portal dropdown (CPU icon, top-right)
- Compare responses from different models
- System automatically detects available models

#### Model Management API
- `GET /models` - List all supported models
- `GET /models/current` - Get currently loaded model
- `GET /models/{model_id}` - Get specific model info
- `POST /models/switch` - Switch to different model (staff only)
- `GET /models/stats` - Get engine statistics

**Files Added:**
- `backend/model_registry.py` - Centralized model configuration

**Files Modified:**
- `backend/multi_model_engine.py` - Dynamic loading/switching
- `backend/main.py` - Model management endpoints
- `frontend/app/staff/page.tsx` - Model selector UI

---

### 📚 Documentation Improvements

#### README.md Comprehensive Update
- **Model comparison table** - All 10 models with specs
- **Download instructions** - 3 methods (quick, multiple, Hugging Face CLI)
- **Filename reference** - Exact filenames expected
- **"What's New" section** - Recent features highlighted
- **Enhanced testing guide** - Step-by-step for all features
- **Model switching instructions** - How to use in staff portal
- **Expanded troubleshooting** - Model-specific issues

#### Self-Contained Documentation
- New developers can onboard completely independently
- No external docs needed
- Clear prerequisites, setup, testing, troubleshooting

---

### 🎨 UI/UX Improvements

#### Staff Portal
- Model selector dropdown with availability indicators
- Colored boxes for reasoning (purple), answer (blue), questions (green)
- Collapsible reasoning section
- Loading states during model switching

#### Prompt Engineering
- Added concrete medical examples in prompts
- Explicit format instructions (REASONING: → ANSWER:)
- Increased token limit from 512 to 768 to prevent truncation

---

## Technical Details

### Dependencies
- **No new dependencies added** - Used Python stdlib only (`html`, `logging`)
- Maintains backward compatibility
- Works offline

### Performance
- HTML escaping: <1ms per request
- Model switching: ~2-5 seconds (depends on model size)
- Multi-section parsing: O(n) where n = text length

### Backward Compatibility
- ✅ DeepSeek `<think>` tags still work
- ✅ Existing API unchanged
- ✅ All previous features maintained

---

## Migration Guide

### For Existing Installations

1. **Pull latest code:**
   ```bash
   git pull origin main
   ```

2. **No dependency changes needed** - using stdlib only

3. **Download additional models (optional):**
   ```bash
   mkdir -p models
   cd models
   # Download any of the 10 supported models
   wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf -O deepseek-r1-1.5b-q4_k_m.gguf
   cd ..
   ```

4. **Restart backend:**
   ```bash
   ./start-backend.sh
   ```

5. **Test new features:**
   - Visit http://localhost:3000/staff
   - Try model switching (top-right dropdown)
   - Ask a question and see 3 colored boxes

---

## Security Audit

| Vulnerability | Severity | Status | Fix |
|--------------|----------|--------|-----|
| XSS Injection | Critical | ✅ Fixed | HTML escaping |
| Silent Failures | High | ✅ Fixed | Logging added |
| Multi-Section Confusion | Medium | ✅ Fixed | Enhanced parsing |

---

## Contributors

This update includes:
- Security hardening (XSS protection, logging)
- Multi-model support (10+ models)
- Chain-of-thought visualization
- Comprehensive documentation
- UI/UX improvements

---

## What's Next

### Planned Features
- [ ] Model performance benchmarking
- [ ] A/B testing different models
- [ ] Custom model fine-tuning guide
- [ ] Docker containerization
- [ ] Model download script

### Known Limitations
- Model switching takes 2-5 seconds (unavoidable due to model loading)
- Some models require GPU for acceptable performance (documented in README)
- Suggested questions extraction may fail on poorly formatted outputs (fallback behavior in place)

---

**For detailed setup instructions, see [README.md](./README.md)**
