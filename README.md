# Team38

# 🚀 RFP Response Analyzer

> **AI-Powered Assistant for Automated RFP Analysis and Compliance Checks**  
> Built with ❤️ using Groq + OLMo + RAG (Retrieval-Augmented Generation)

---

## 🧠 Problem Statement

ConsultAdd collaborates with U.S. government agencies and must regularly respond to complex **Requests for Proposals (RFPs)**. These documents are often long, legal-heavy, and require precise compliance checks. Currently, the review process is:

- 🔍 Manual  
- 🕒 Time-consuming  
- ⚠️ Prone to human error  

The **challenge** is to build a solution that **automates RFP analysis**, reduces effort, ensures compliance, and flags risks using **Generative AI**, **RAG**, and **Agentic Workflows**.

---

## 💡 Our Solution: AI-Powered RFP Analyzer Chatbot

We’ve developed a smart document analysis tool that leverages **Groq’s blazing fast inference** and **OLMo's open-source LLM** capabilities to automate:

### ✅ 1. Compliance Checks

- Instantly verifies **ConsultAdd's eligibility** (e.g., certifications, registrations)
- Flags **deal-breaker clauses** or missing qualifications

### 📌 2. Eligibility Criteria Extraction

- Summarizes key qualifications and certifications
- Provides quick insights into whether the RFP is worth pursuing

### 📝 3. Submission Checklist Generation

- Auto-generates a checklist of:
  - Page limits, font styles, TOC
  - Mandatory forms and attachments
- Ensures **nothing is missed** in the submission process

### ⚖️ 4. Contract Risk Analysis

- Identifies biased or risky contract clauses (e.g., one-sided terminations)
- Suggests **modifications** to ensure fair legal standing

---

## 🧰 Tech Stack

| Layer          | Tech                     |
|----------------|--------------------------|
| **Frontend**   | HTML , CSS, jS   |
| **Backend**    | Flask                    |
| **LLM & RAG**  |  FAISS , Ollama                  |
| **PDF Parsing**| LangChain + PyPDF2       |
| **Deployment** | Localhost / Cloud Ready  |

---

## 🧪 Features in Action

> 🔄 Upload an RFP → 📂 Parse → 📋 Extract Info → 🛡️ Flag Risks → ✅ Final Checklist  
All in **seconds**, thanks to **Ollama ultra-fast inference**.

---

## 🧠 Why  Ollama?

- ⚡ **OLLAMA*: Industry-leading inference speed. Perfect for real-time chat and multi-turn agent workflows.
-      Open and transparent, optimized for legal and government tasks.
- 🔍 **RAG**: Grounded answers, fewer hallucinations.

---

## 📁 Folder Structure

├── main.py # Entry point ├── rfp_parser.py # PDF & eligibility extractor ├── checklist_generator.py # Submission requirements generator ├── risk_analyzer.py # Contract clause analysis ├── chatbot_agent.py # Conversational agent logic ├── .env # Secure API key storage ├── README.md # You're reading it!

