# Financial Intelligence Research Agent

An AI-powered Financial Research Assistant that combines Retrieval-Augmented Generation (RAG), financial valuation models, sentiment analysis, and structured financial reasoning to generate grounded equity research reports.

The long-term objective of this project is to evolve into a domain-specific Financial Small Language Model (SLM) capable of performing reliable investment research with minimal hallucinations.

---

## Features

### Financial Statement Analysis

- Automatic financial statement normalization
- Revenue, EBIT and Net Income growth analysis
- Revenue CAGR calculation
- Free Cash Flow analysis
- Operating Margin
- Net Margin
- ROE
- EPS
- Debt-to-Equity
- Financial health interpretation

---

### Valuation Engine

- Discounted Cash Flow (DCF)
- FCFF Valuation
- WACC Calculation
- Enterprise Value
- Equity Value
- Intrinsic Value Estimation

---

### AI Research Pipeline

- Retrieval-Augmented Generation (RAG)
- ChromaDB Vector Database
- Cross-Encoder Reranking
- Query Intent Classification
- Intelligent Retrieval Planning
- Structured Prompt Generation
- Context-aware Report Generation

---

### Earnings Call Intelligence

- Transcript Ingestion
- Semantic Chunking
- Financial Section Detection
- Speaker Identification
- Evidence Extraction
- Citation-ready Context Building

---

### Sentiment Analysis

- FinBERT-based Financial Sentiment Analysis
- Confidence Scoring
- Earnings Call Sentiment Summary

---

### Evaluation Framework

- Retrieval Accuracy
- Grounding Score
- Citation Coverage
- Completeness
- Latency Tracking
- Benchmark Dataset Support

---

## Current Architecture

```
User Query
     │
     ▼
Query Classifier
     │
     ▼
Retrieval Planner
     │
     ▼
Financial Analysis Engine
     │
     ▼
Vector Retrieval (ChromaDB)
     │
     ▼
Cross Encoder Reranker
     │
     ▼
Evidence Builder
     │
     ▼
Prompt Builder
     │
     ▼
LLM
     │
     ▼
Investment Research Report
```

---

## Tech Stack

### AI

- Qwen 2.5
- LangChain
- ChromaDB
- SentenceTransformers
- Cross Encoder Reranker
- FinBERT

### Financial Analytics

- Discounted Cash Flow (DCF)
- FCFF
- WACC
- Financial Ratio Analysis

### Backend

- Python
- Pandas
- NumPy
- yFinance

### Interface

- Streamlit

---

## Repository Structure

```
app/

analysis/
core/
data/
evaluation/
nlp/
rag/
reasoning/
valuation/

benchmarks/

streamlit_app.py
```

---

## Current Capabilities

- Financial statement normalization
- Company financial analysis
- Intelligent evidence retrieval
- Earnings call understanding
- AI-powered investment reports
- Valuation modelling
- Financial sentiment analysis
- Benchmark evaluation

---

## Benchmarked Companies

- Apple
- Microsoft
- Amazon
- NVIDIA
- Tesla

---

## Sample Research Output

The system generates:

- Executive Summary
- Financial Intelligence
- Valuation Summary
- Sentiment Analysis
- Earnings Call Evidence
- AI Investment Analysis

---

## Roadmap

### Completed

- Financial Statement Normalization
- Financial Analysis Engine
- DCF Valuation
- FCFF Engine
- WACC Engine
- ChromaDB Integration
- Cross Encoder Reranking
- Query Intent Classification
- Retrieval Planner
- FinBERT Integration
- Benchmark Framework

---

### In Progress

- Prompt Optimization
- Grounded Citation Generation
- Latency Optimization
- Multi-company Benchmark Expansion
- Real Earnings Call Dataset

---

### Planned

- Hybrid Retrieval (Vector + BM25)
- Multi-quarter Financial Reasoning
- Automated Evaluation Dashboard
- Portfolio Analysis
- Financial Knowledge Graph
- Financial SLM Fine-tuning
- Autonomous Financial Research Agent

---

## Long-Term Vision

This project is being developed as a production-quality Financial AI system capable of evolving into a Financial Small Language Model (SLM).

The focus is on:

- High factual accuracy
- Grounded reasoning
- Explainable financial analysis
- Minimal hallucinations
- Modular AI architecture
- Enterprise-ready financial intelligence

---

## Author

**Shivaum Sharma**

Computer Science Engineering (Data Science)

Manipal Institute of Technology
