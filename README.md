# 🎬 AI Video Assistant (Meeting Intelligence & RAG)

> An end-to-end multimodal AI platform that extracts audio from YouTube videos or local files, performs bilingual transcription (English & Hinglish), generates structured meeting summaries and action items, and powers an interactive RAG conversational assistant.

---

## 🌟 Key Features

- **Multimodal Ingestion**: Accepts YouTube URLs or local video/audio files (`.mp4`, `.mov`, `.wav`, `.mp3`).
- **Bilingual Speech-to-Text**:
  - **English**: Local OpenAI Whisper inference running on PyTorch.
  - **Hinglish / Hindi**: Slices audio and routes to Sarvam AI (`saaras:v2.5`) for simultaneous transcription and translation to English.
- **Meeting Intelligence (Mistral AI & LangChain LCEL)**:
  - **Executive Title**: Generated from meeting context (max 8 words).
  - **Map-Reduce Summary**: Chunks transcripts and recursively summarizes them into an executive bullet-point brief.
  - **Action Items**: Extracts tasks, owners, and deadlines.
  - **Key Decisions & Open Questions**: Identifies consensus items and follow-up questions.
- **Semantic Search & RAG Chat**:
  - Embedded vector search using HuggingFace (`all-MiniLM-L6-v2`) and local ChromaDB.
  - Grounded RAG chain preventing hallucinations with strict context checks.
- **Dual User Interfaces**:
  - **Streamlit Web Dashboard**: Modern dark-mode interface with live status tracker, expandable transcript viewer, and chat UI.
  - **CLI Terminal**: Fast command-line runner and REPL.

---

## 🏗️ System Architecture

```
[YouTube URL / Local Media]
            │
            ▼
   utils/audio_processor.py (yt-dlp, FFmpeg 16kHz mono, 10-min chunking)
            │
            ▼
   core/transcriber.py
   ├── English   ──► Local OpenAI Whisper (small model)
   └── Hinglish  ──► Sarvam AI (25s slices -> English)
            │
            ▼ (Full Transcript)
  ┌─────────┴────────────────────────┐
  ▼                                  ▼
core/summarizer.py & extractor.py    core/vector_store.py
• Title Generation                   • Recursive Text Splitter (500 chars)
• Map-Reduce Summarizer              • HuggingFace all-MiniLM-L6-v2
• Action Items (Task, Owner, Due)    • ChromaDB local store (./vector_db)
• Key Decisions & Questions                  │
  │                                          ▼
  │                                  core/rag_engine.py
  │                                  • Top-4 Similarity Retrieval
  │                                  • Mistral LCEL RAG Chain
  └───────────────────┬──────────────────────┘
                      ▼
         [main.py (CLI) / app.py (Streamlit Web UI)]
```

---

## 📁 Project Structure

```
AI Video Assistant/
├── core/
│   ├── __init__.py
│   ├── extractor.py         # LCEL chains for Action Items, Decisions, Questions
│   ├── rag_engine.py        # LangChain RAG pipeline & strict Q&A prompt
│   ├── summarizer.py        # Map-Reduce summarization & title generation
│   ├── transcriber.py       # Whisper & Sarvam AI transcription routing
│   └── vector_store.py      # ChromaDB setup and HuggingFace embeddings
├── utils/
│   ├── __init__.py
│   └── audio_processor.py   # yt-dlp download, WAV conversion, audio chunking
├── app.py                   # Streamlit web application
├── main.py                  # Command-line interface runner
├── Requirements.txt         # Project dependencies
├── .env.example             # Example environment variables template
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python >= 3.10**
- **FFmpeg**: Must be installed and added to your system `PATH`.
  - On Windows: Install via `winget install Gyan.FFmpeg` or download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add to PATH.
  - Test with: `ffmpeg -version`

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd "AI Video Assistant"

# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r Requirements.txt
```

### 3. Environment Setup
Copy `.env.example` to `.env` and insert your API keys:
```bash
cp .env.example .env
```

Edit `.env`:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here    # Required only for Hinglish mode
WHISPER_MODEL=small                         # Optional: tiny, base, small, medium, large
```

---

## 💻 Running the Application

### Option A: Web Dashboard (Streamlit)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Paste a YouTube URL or local file path, select the language, and click **Analyse**.

### Option B: Terminal CLI
```bash
python main.py
```
Follow the interactive prompts to process audio and chat with the transcript in your terminal.

---

## 🧠 Technical Highlights for Interviews

1. **Map-Reduce Summarization Pattern**: Instead of naive prompt stuffing, transcripts are chunked recursively, summarized independently (Map), and synthesized into a structured brief (Reduce).
2. **Hallucination Prevention**: Strict negative prompt constraints combined with low LLM temperature (`0.2` - `0.3`) ensure the assistant answers strictly from retrieved context.
3. **Smart Audio Slicing**: Sarvam AI has a 30-second API limit. The audio processor splits chunks into 25-second pieces with automated disk cleanup to ensure zero payload rejections.
4. **Clean Decoupled Architecture**: Logic is separated into clear modules (`utils/audio_processor`, `core/transcriber`, `core/summarizer`, `core/rag_engine`), making it testable and maintainable.
