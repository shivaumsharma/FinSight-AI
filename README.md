# Finsight AI – Autonomous Financial Research Agent

Finsight AI is an AI-powered financial research platform that combines Retrieval-Augmented Generation (RAG), financial statement analysis, Discounted Cash Flow (DCF) valuation, earnings call analysis, and investment thesis generation into a unified research workflow.

The project is designed to automate institutional-style equity research by integrating financial data retrieval, valuation models, semantic search, and natural language generation.

---

## Features

- Financial statement analysis using live market data
- Discounted Cash Flow (DCF) valuation
- Weighted Average Cost of Capital (WACC) calculation
- Free Cash Flow to Firm (FCFF) forecasting
- Earnings call transcript retrieval using RAG
- Semantic search with ChromaDB
- Financial statement normalization
- AI-generated research reports
- Investment thesis generation
- Sensitivity analysis for valuation assumptions

---

## Architecture

```
                     User
                       │
                       ▼
                 Streamlit UI
                       │
                       ▼
              Financial Research Engine
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   Market Data      RAG Engine    Valuation Engine
        │              │              │
        ▼              ▼              ▼
   Financials     ChromaDB +      FCFF → WACC →
                  Earnings Calls      DCF
        │              │              │
        └──────────────┼──────────────┘
                       ▼
             Investment Thesis Generator
                       │
                       ▼
                Financial Report
```

---

## Project Structure

```
app/
│
├── analysis/
│   └── investment_thesis.py
│
├── data/
│   ├── market_data.py
│   ├── financial_normalizer.py
│   └── earnings transcripts
│
├── nlp/
│   └── sentiment.py
│
├── rag/
│   ├── rag_pipeline.py
│   ├── chroma_store.py
│   ├── transcript_loader.py
│   ├── text_chunker.py
│   └── report_generator.py
│
├── valuation/
│   ├── fcff_engine.py
│   ├── wacc_engine.py
│   ├── dcf_engine.py
│   ├── sensitivity_engine.py
│   └── valuation_pipeline.py
│
streamlit_app.py
requirements.txt
runtime.txt
```

---

## Core Components

### Market Data

- Live company financial statements
- Balance Sheet
- Income Statement
- Cash Flow Statement
- Company information

Powered by **Yahoo Finance (yfinance)**.

---

### Retrieval-Augmented Generation (RAG)

- Earnings call transcript ingestion
- Text chunking
- Sentence embeddings
- ChromaDB vector storage
- Semantic retrieval
- AI-powered report generation

---

### Financial Valuation

The valuation engine performs:

- Revenue forecasting
- FCFF forecasting
- Tax rate estimation
- Net Operating Profit After Tax (NOPAT)
- Working capital adjustments
- WACC calculation
- Enterprise Value estimation
- Equity Value estimation
- Intrinsic Value calculation
- Sensitivity Analysis

---

### Investment Thesis

Generates:

- Bullish indicators
- Bearish indicators
- Valuation summary
- Buy / Hold / Sell recommendation

---

## Technologies Used

### Backend

- Python
- Pandas
- NumPy

### Machine Learning

- Transformers
- Sentence Transformers
- LangChain

### Vector Database

- ChromaDB

### Financial Data

- yfinance

### Frontend

- Streamlit

### NLP

- FinBERT
- FLAN-T5

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run streamlit_app.py
```

---

## Current Workflow

1. Enter a stock ticker
2. Load live financial statements
3. Retrieve earnings call transcript
4. Build vector database
5. Ask financial questions
6. Retrieve relevant transcript context
7. Generate AI response
8. Run DCF valuation (if required)
9. Generate investment thesis
10. Display final report

---

## Roadmap

### Completed

- Financial statement ingestion
- DCF valuation engine
- WACC engine
- FCFF forecasting
- Financial statement normalization
- Earnings call RAG
- ChromaDB integration
- Streamlit dashboard
- Investment thesis generation

### In Progress

- Multi-agent financial reasoning
- SEC filing ingestion
- News intelligence
- Portfolio analysis
- Advanced report generation
- Improved financial reasoning
- Production backend architecture

### Future Vision

The long-term objective of Finsight AI is to evolve into an autonomous financial intelligence platform capable of institutional-grade equity research.

Future releases will include:

- Financial AI Agent
- Tool-based reasoning
- Portfolio optimization
- Multi-company comparison
- SEC filing analysis
- Real-time news intelligence
- Financial Small Language Model (SLM)
- Autonomous investment research workflows

---

## Disclaimer

This project is intended for educational and research purposes only and should not be considered financial advice or investment recommendations.