# RAG Chatbot

A PDF-grounded chatbot built with LangGraph, hybrid retrieval (BM25 + FAISS), and cross-encoder reranking - plus live web search, a self-built MCP expense tracker server, stock prices, and a calculator, all wired into a single tool-calling agent.

## Features

### 1. Chat + Live Web Search (Tavily)

General conversation with access to real-time web search when a question needs current information beyond the uploaded document.

![Chat with Tavily web search](screenshots/chat-tavily.png)
_The agent answers a general question, then pulls and summarizes current sports news via Tavily web search into a structured table._

### 2. MCP-Based Expense Tracker

A custom-built Model Context Protocol (MCP) server for expense tracking, connected to the agent over stdio transport - add expenses conversationally, then query them back as structured tables.

![Adding an expense](screenshots/expense-tracker1.png)
_Adding a new expense conversationally ("add 10,000 to my expenses, label them as clothes") - the MCP server parses this into a structured entry and confirms it back._

![Viewing full expense history](screenshots/expense-tracker2.png)
_Requesting the complete expense history - the MCP server returns all recorded entries, rendered as a table with date, amount, category, sub-category, and notes._

### 3. RAG over PDFs

Upload a PDF and ask questions grounded in its actual content - hybrid search (BM25 + FAISS) retrieves relevant chunks, a cross-encoder reranks them, and answers are generated with page-level citations.

![RAG answer with citations from an uploaded PDF](screenshots/rag-pdf.png)
_Asking for a summary of Java operators with code snippets from an uploaded 104-page PDF - the answer is generated from retrieved chunks and cited back to the exact source page._

## RAG Pipeline

The retrieval-augmented generation pipeline combines several techniques rather than relying on plain vector similarity search alone:

**1. Ingestion & Chunking**

- PDFs are loaded page-by-page via `PyPDFLoader`.
- Pages are merged into one continuous text stream _before_ splitting, with page-boundary markers inserted between them. This avoids a common failure mode where a sentence spanning two PDF pages gets truncated mid-sentence, since naive per-page chunking splits each page's text independently and can never merge content across the page break.
- `RecursiveCharacterTextSplitter` (chunk size 900, overlap 150, sentence-aware separators) splits the merged text, and each resulting chunk is re-tagged with the correct page number recovered from the nearest marker.

**2. Hybrid Retrieval**

- **Sparse (BM25):** keyword-based retrieval over the chunk corpus. Uses a custom lowercasing `preprocess_func`, since BM25's default tokenizer is case-sensitive and will silently fail to match a query like "coffee house" against a chunk containing the proper noun "Coffee House."
- **Dense (FAISS + HuggingFace embeddings, `BAAI/bge-small-en-v1.5`):** semantic similarity search using Maximal Marginal Relevance (MMR) rather than plain top-k similarity, to reduce near-duplicate chunks crowding out more relevant but differently-worded results.
- **Dual-query search:** both the user's raw query and a HyDE-expanded version of it are run through both retrievers, and the results are merged into one deduplicated candidate pool. This exists because HyDE's generated hypothetical passage is sometimes too generic to mention a specific named entity the raw query already contains.

**3. HyDE (Hypothetical Document Embeddings)**
Before retrieval, the LLM generates a short hypothetical passage that would answer the query, and that hypothetical text is embedded and searched alongside the raw query. Hypothetical answers tend to be phrased closer to how the source document itself is written than a short user question is, which improves embedding-similarity matches - particularly for vague or under-specified queries.

**4. Reranking**
The merged candidate pool from both retrievers is reranked using a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which scores each candidate directly against the _original_ (non-HyDE) query text for final relevance, since cross-encoders are more accurate at judging query-document relevance than the bi-encoder similarity scores used for initial retrieval.

**5. Answer Generation & Citation**
The reranked context is passed to the LLM to generate a natural-language answer. Citations are built from the actual retrieved chunk metadata (filename + page number), not left to the model to state on its own.

## MCP Expense Tracker

One of the agent's tools is a **Model Context Protocol (MCP) server built from scratch** for expense tracking, rather than a plain Python function tool. It runs as a separate local process and communicates with the main agent over **stdio transport**, using `FastMCP` on the server side and `langchain-mcp-adapters` (`MultiServerMCPClient`) on the client side to expose its tools to the LangGraph agent.

Why MCP instead of a regular tool function: it demonstrates a standardized, provider-agnostic way to expose external capabilities to an LLM agent - the same MCP server could in principle be reused by any MCP-compatible client, not just this specific LangGraph app.

**Engineering notes:**

- MCP tools loaded via `langchain-mcp-adapters` are async-only (`ainvoke`, no `invoke`), but LangGraph's `ToolNode` calls tools synchronously in this app's execution mode. Each async MCP tool is wrapped with a synchronous entrypoint that schedules its coroutine onto a persistent background event loop (created once at startup) and blocks for the result - avoiding the overhead of spinning up a new event loop on every single tool call.
- The server process is launched via a direct path to its own virtual environment's Python interpreter, keeping its dependencies isolated from the main app's environment.

## Tech Stack

- **Orchestration:** LangChain, LangGraph
- **LLM:** Groq (`openai/gpt-oss-20b`)
- **Retrieval:** BM25 (sparse) + FAISS (dense) combined via Reciprocal Rank Fusion, with Maximal Marginal Relevance for diversity
- **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
- **Reranking:** Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Query expansion:** HyDE (Hypothetical Document Embeddings)
- **Tools:** self-built MCP expense tracker server (FastMCP + stdio transport), Tavily web search, Alpha Vantage stock prices, calculator
- **Frontend:** Streamlit, with per-thread conversation history via LangGraph's checkpointer

## Key Engineering Details

- **Page-boundary-safe chunking** - PDF pages are merged into one continuous text before splitting, so sentences that happen to span a page break aren't truncated mid-sentence (a real, previously-fixed retrieval bug).
- **Case-insensitive BM25 matching** - the default tokenizer is case-sensitive, which silently drops matches on proper nouns; fixed via a custom `preprocess_func`.
- **Dual-query retrieval** - both the raw query and its HyDE expansion are searched independently and merged, since HyDE's generic hypothetical text can miss specific named entities.
- **Multi-conversation threads** - each chat thread has its own isolated document index and history, with a thread-scoped file uploader to prevent cross-thread document leakage.
- **Async MCP tools in a sync graph** - a persistent background event loop bridges LangGraph's synchronous tool execution with the async-only MCP tool interface.

## Setup

```bash
git clone https://github.com/Raghav-012/RAG_chatbot.git
cd RAG_chatbot
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
ALPHAVANTAGE_API_KEY=your_key_here
```

Run the app:

```bash
streamlit run frontend.py
```

## License

MIT
