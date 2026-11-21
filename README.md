# 📄 PDF RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Runtime-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GitHub](https://img.shields.io/badge/GitHub-nishali26-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nishali26/PDF-rag-Chatbot)

An interactive **Retrieval-Augmented Generation (RAG)** application written in Python (`app.py`) that ingests PDF documents, extracts their textual content, and answers user questions grounded in the document context—supporting local **HuggingFace (`google/flan-t5-small`)** inference via `transformers` alongside optional external LLM API workflows.

---

## 📋 Table of Contents

- [Overview & Problem Statement](#-overview--problem-statement)
- [Key Features](#-key-features)
- [System Architecture & RAG Pipeline](#-system-architecture--rag-pipeline)
- [Detailed Tech Stack](#-detailed-tech-stack)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Local Setup & Installation](#-local-setup--installation)
- [Usage Guide](#-usage-guide)
- [Security & `.gitignore` Practices](#-security--gitignore-practices)
- [Troubleshooting & Common Issues](#-troubleshooting--common-issues)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 🎯 Overview & Problem Statement

Standard large language models rely on general parametric training data and can hallucinate or guess when asked about specific, private, or newly uploaded PDFs—such as research papers, technical manuals, university notes, or reports. Furthermore, relying strictly on paid cloud LLM APIs during development and testing can introduce rate limits, quota blocks, or recurring costs.

**PDF RAG Chatbot** provides a practical Python workflow (`app.py`) to query PDF content directly and run local inference with HuggingFace's [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) model so you can test, iterate, and inspect document-grounded responses without hitting external API quota walls.

---

## ✨ Key Features

- **📑 PDF Document Ingestion & Text Extraction:** Reads PDF files and extracts readable text so questions are anchored to actual document content rather than guesswork.
- **🤖 Grounded Question Answering (RAG Workflow):** Combines extracted PDF context with user queries inside structured prompts to keep answers aligned with the source file.
- **🤗 Local HuggingFace Model Execution:** Uses [`transformers`](https://huggingface.co/docs/transformers/index) with `google/flan-t5-small` for local text-to-text generation—ideal when external API quotas are exhausted or offline/local experimentation is preferred.
- **🔌 Extensible LLM & Environment Configuration:** Supports `.env` environment variables (such as `OPENAI_API_KEY` or model configuration flags) so you can adapt or extend the pipeline cleanly.
- **⚡ Developer-Friendly Local Setup:** Designed for fast iteration on Windows (PowerShell), macOS, and Linux inside an isolated Python virtual environment.

---

## 🏗️ System Architecture & RAG Pipeline

```text
+-----------------------------------------------------------------------+
|                           INPUT LAYER                                 |
|   [ PDF File (.pdf) ]                      [ User Question / Query ]  |
+------------------+----------------------------------------+-----------+
                   |                                        |
                   v                                        |
+-----------------------------------+                       |
|      PDF PARSING & TEXT EXTRACTION|                       |
|  Extract raw text & clean pages   |                       |
+------------------+----------------+                       |
                   |                                        |
                   v                                        |
+-----------------------------------+                       |
|   CONTEXT PREPARATION & PROMPTING | <---------------------+
|  Assemble document context + Q&A  |
|  prompt for grounded generation   |
+------------------+----------------+
                   |
                   v
+-----------------------------------------------------------------------+
|                        INFERENCE ENGINE                               |
|  +-----------------------------------------------------------------+  |
|  | Local HuggingFace Pipeline (transformers + torch)               |  |
|  | Model: google/flan-t5-small (zero external quota bottlenecks)   |  |
|  +-----------------------------------------------------------------+  |
|  Optional / Extensible: External LLM APIs via .env configuration    |
+------------------+----------------------------------------------------+
                   |
                   v
+-----------------------------------------------------------------------+
|                         OUTPUT & RESPONSE                             |
|       Grounded answer returned to user via CLI / app interface        |
+-----------------------------------------------------------------------+