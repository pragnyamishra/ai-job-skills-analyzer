# AI Job Skills Analyzer

An autonomous LLM-powered platform that analyzes real-time job market data, identifies your skill gaps, and generates a personalized learning plan.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                            │
│   Resume Upload  →  Role Selection  →  Gap Report  →  Plan     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  LangGraph Agent │
                    │  (Orchestrator)  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐     ┌───────▼───────┐    ┌──────▼──────┐
    │ JSearch  │     │   Groq API    │    │  Pinecone   │
    │ RapidAPI │     │ (Llama 3.3)   │    │ Vector DB   │
    │          │     │               │    │             │
    │ Live JDs │     │ Skill NER +   │    │ Embeddings  │
    │ Fetcher  │     │ Plan Gen      │    │ + Matching  │
    └──────────┘     └───────────────┘    └─────────────┘
                             │
                    ┌────────▼────────┐
                    │ Langfuse + Logs │
                    │ (LLMOps Layer)  │
                    └─────────────────┘
```

## Pipeline Flow

```
User inputs resume + target role
        ↓
1. Fetch 25-50 live job postings (RapidAPI JSearch)
        ↓
2. Extract + normalize skills from all JDs (Groq, few-shot prompting)
        ↓
3. Embed market skills in Pinecone (sentence-transformers)
        ↓
4. Extract skills from resume (Groq)
        ↓
5. Semantic gap analysis (Pinecone cosine similarity)
        ↓
6. Generate personalized learning plan (Groq)
        ↓
Dashboard: match score, gap report, week-by-week plan
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (autonomous multi-step agent) |
| LLM Inference | Groq API (Llama 3.3 70B) |
| Prompt Engineering | Few-shot, chain-of-thought |
| Vector Database | Pinecone (cosine similarity) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| NER Fine-tuning | HuggingFace Transformers (DistilBERT) |
| Job Data | RapidAPI JSearch |
| LLMOps Monitoring | Langfuse + Python logging |
| Frontend | Streamlit |
| Visualization | Plotly |

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourusername/ai-job-skills-analyzer.git
cd ai-job-skills-analyzer
pip install -r requirements.txt
```

### 2. Get API keys

| Key | Where | Free Tier |
|-----|-------|-----------|
| GROQ_API_KEY | [console.groq.com](https://console.groq.com) | Yes |
| RAPID_API_KEY | [rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) | 500 req/month |
| PINECONE_API_KEY | [app.pinecone.io](https://app.pinecone.io) | Yes (1 index) |
| LANGFUSE keys | [cloud.langfuse.com](https://cloud.langfuse.com) | Yes (optional) |

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual keys
```

### 4. Run

```bash
streamlit run app.py
```

## Fine-tuning (Optional)

Train a custom NER model for more accurate skill extraction:

```bash
# Step 1: Generate training data
python fine_tune.py --generate-data --samples 200

# Step 2: Train model
python fine_tune.py --train

# Step 3: Test
python fine_tune.py --test "Experience with PyTorch, Docker, and AWS required"
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Add API keys under Settings → Secrets:
   ```toml
   GROQ_API_KEY = "your_key"
   RAPID_API_KEY = "your_key"
   PINECONE_API_KEY = "your_key"
   ```
5. Deploy

## Project Structure

```
ai-job-skills-analyzer/
├── app.py                  # Streamlit frontend
├── agent.py                # LangGraph orchestration
├── job_fetcher.py           # RapidAPI JSearch integration
├── skill_extractor.py       # Groq-powered skill NER
├── vector_store.py          # Pinecone embeddings + similarity
├── resume_parser.py         # PDF/DOCX/TXT parser
├── gap_analyzer.py          # Skill gap computation
├── learning_plan.py         # Learning path generator
├── monitoring.py            # Langfuse + logging (LLMOps)
├── fine_tune.py             # NER fine-tuning script
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml          # Streamlit theme
└── data/
    └── api_calls.log        # Local monitoring log
```
