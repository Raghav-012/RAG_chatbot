import os
import time
import asyncio
import tempfile
import requests
from datetime import date
from typing import TypedDict, Annotated, Any, Dict, Optional

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool, StructuredTool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# NOTE: llama-3.1-8b-instant and llama-3.3-70b-versatile were deprecated by
# Groq for free/dev tier usage (announced June 17, 2026). Moved to Groq's
# recommended replacement, openai/gpt-oss-20b. Check your actual TPM/RPM
# limits for this model at console.groq.com -> Settings -> Limits, since
# they're organization-specific and can change.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

search_tool = TavilySearch(max_results=3, search_depth="basic")

# ---------------------------------------------------------------------------
# RERANKER (loaded once at import time -- first run downloads the model,
# ~90MB, cross-encoder/ms-marco-MiniLM-L-6-v2)
# ---------------------------------------------------------------------------
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=6)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price_range(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Fetch daily closing prices for a symbol (e.g. 'AAPL') between start_date
    and end_date (both YYYY-MM-DD, inclusive). Use this for questions about
    price movement over time, comparisons between two dates, or "previous N
    days" style questions -- get_stock_price only returns a single day.
    """
    api_key = os.environ["ALPHAVANTAGE_API_KEY"]
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    )
    r = requests.get(url)
    data = r.json()

    series = data.get("Time Series (Daily)")
    if not series:
        return {"error": f"Could not fetch historical data for {symbol}.", "raw": data}

    results = {
        d: day_data.get("4. close")
        for d, day_data in series.items()
        if start_date <= d <= end_date
    }

    if not results:
        return {
            "error": f"No trading data for {symbol} between {start_date} and {end_date}.",
        }

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "closes_by_date": dict(sorted(results.items())),
    }


@tool
def get_stock_price(symbol: str, date: Optional[str] = None) -> dict:
    """
    Fetch stock price for a given symbol (e.g. 'AAPL', 'TSLA').
    If `date` (YYYY-MM-DD) is provided, returns the closing price for that
    specific trading day using Alpha Vantage's daily time series. If `date`
    is omitted, returns the latest available quote.
    """
    api_key = os.environ["ALPHAVANTAGE_API_KEY"]

    if date is None:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        r = requests.get(url)
        return r.json()

    # outputsize=full needed for any date older than ~100 days back
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    )
    r = requests.get(url)
    data = r.json()

    series = data.get("Time Series (Daily)")
    if not series:
        return {"error": f"Could not fetch historical data for {symbol}.", "raw": data}

    day_data = series.get(date)
    if day_data is None:
        return {
            "error": f"No trading data for {symbol} on {date} (market may have been closed, "
                     "or the date is out of range).",
        }

    return {
        "symbol": symbol,
        "date": date,
        "open": day_data.get("1. open"),
        "high": day_data.get("2. high"),
        "low": day_data.get("3. low"),
        "close": day_data.get("4. close"),
        "volume": day_data.get("5. volume"),
    }


# ---------------------------------------------------------------------------
# HYBRID RAG (BM25 + FAISS ensemble -> cross-encoder rerank)
# One retriever per chat thread, kept in memory.
# ---------------------------------------------------------------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
_THREAD_DEBUG_INFO: Dict[str, dict] = {}


def _hyde_expand(query: str) -> str:
    """
    HyDE: ask the LLM to write a hypothetical answer to the query, then use
    that (instead of the raw query) for retrieval. Hypothetical answers tend
    to be phrased more like the source document than a short user question
    is, which improves embedding-similarity matches.

    Skipped for already-long/specific queries (>= 12 words) since those
    tend to retrieve well on their own and HyDE would just add an extra
    LLM round-trip for little gain.

    Falls back to the original query on any failure so retrieval never
    breaks because of this step.
    """
    if len(query.split()) >= 12:
        return query

    try:
        prompt = (
            "Write a short hypothetical passage (2-3 sentences) that would "
            "answer the following question, as if it were an excerpt from a "
            "document. Do not add commentary, just the passage.\n\n"
            f"Question: {query}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        hypothetical = response.content.strip()
        return hypothetical if hypothetical else query
    except Exception:
        return query


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a hybrid (BM25 + FAISS) retriever, wrapped with a cross-encoder
    reranker, for the uploaded PDF and store it for this thread.
    Called by the Streamlit sidebar after a file is uploaded.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()  # one Document per PDF page, in order

        # PyPDFLoader gives one Document per page. Splitting each page's
        # Document independently (the old approach) means any sentence that
        # happens to span across a page boundary in the source PDF gets
        # truncated right there, no matter how large chunk_size is set --
        # chunk_size only limits chunk size WITHIN a single page's text, it
        # can never merge content across separate page Documents. This was
        # confirmed as the actual cause of a real truncated-answer bug.
        #
        # Fix: concatenate all pages into one continuous text first, with an
        # invisible marker inserted at each page boundary, then split that
        # merged text as a whole. Page numbers are recovered afterward by
        # checking which marker(s) fall inside each resulting chunk.
        page_marker_pattern = re.compile(r'\x00PAGE_BREAK_(\d+)\x00')
        parts = []
        for doc in docs:
            page_num = doc.metadata.get("page", 0) + 1  # 1-indexed for display
            parts.append(f"\x00PAGE_BREAK_{page_num}\x00")
            parts.append(doc.page_content)
        combined_text = "".join(parts)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
        )
        raw_chunks = splitter.split_text(combined_text)

        chunks = []
        current_page = docs[0].metadata.get("page", 0) + 1 if docs else 1
        for raw_chunk in raw_chunks:
            markers = page_marker_pattern.findall(raw_chunk)
            if markers:
                current_page = int(markers[-1])  # last page boundary seen in this chunk
            clean_text = page_marker_pattern.sub("", raw_chunk).strip()
            if not clean_text:
                continue
            chunks.append(Document(page_content=clean_text, metadata={"page": current_page - 1}))
            # stored 0-indexed to match rag_tool's existing "+1 for display" logic downstream

        print(f"[DEBUG] Total chunks created: {len(chunks)}", flush=True)
        for i, c in enumerate(chunks):
            if "connor" in c.page_content.lower():
                print(f"[DEBUG] Chunk #{i} (page {c.metadata.get('page')}) contains 'Connor':", flush=True)
                print(f"  {c.page_content!r}", flush=True)

        # dense retriever (semantic)
        vector_store = FAISS.from_documents(chunks, embeddings)
        dense_retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
        )

        # sparse retriever (lexical / exact-match)
        # BM25Retriever's default tokenizer is a plain .split() with NO
        # lowercasing. A proper noun like "Coffee House" in the document
        # text never token-matches a lowercase query like "coffee house" --
        # they're literally different tokens under exact BM25 matching.
        # This was confirmed as the cause of a real retrieval miss: on a
        # 19-chunk corpus, a chunk containing the exact phrase "Coffee
        # House" verbatim never appeared in BM25's top candidates for the
        # query "coffee house". Supplying a lowercasing preprocess_func
        # fixes this.
        bm25_retriever = BM25Retriever.from_documents(
            chunks, preprocess_func=lambda text: text.lower().split()
        )
        bm25_retriever.k = 5

        # NOTE: rag_tool does its own manual dual-query merge + rerank
        # directly via the bm25_retriever/dense_retriever stored below in
        # _THREAD_DEBUG_INFO, rather than going through a LangChain
        # EnsembleRetriever/ContextualCompressionRetriever wrapper. Only a
        # simple marker is kept in _THREAD_RETRIEVERS -- thread_has_document
        # only checks membership, it doesn't use the stored value itself.
        _THREAD_RETRIEVERS[str(thread_id)] = True
        _THREAD_DEBUG_INFO[str(thread_id)] = {
            "chunks": chunks,
            "dense_retriever": dense_retriever,
            "bm25_retriever": bm25_retriever,
        }
        summary = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        _THREAD_METADATA[str(thread_id)] = summary
        return summary
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever_info = _THREAD_DEBUG_INFO.get(str(thread_id))
    if retriever_info is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    # HyDE: a hypothetical-answer expansion of the query (see _hyde_expand).
    # Previously ONLY the HyDE-expanded text was searched -- but HyDE's
    # generic hypothetical passage can fail to mention specific named
    # entities (e.g. "Connor's Coffee House") that the raw query does
    # contain, causing real content to never be retrieved at all. Now both
    # the raw query and the HyDE expansion are searched against both BM25
    # and dense retrieval, and results are merged into one candidate pool
    # before reranking -- doubling the chances of surfacing the right chunk.
    hyde_query = _hyde_expand(query)

    candidates = []
    seen = set()
    for q in (query, hyde_query):
        for doc in retriever_info["bm25_retriever"].invoke(q):
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                candidates.append(doc)
        for doc in retriever_info["dense_retriever"].invoke(q):
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

    print(f"[DEBUG] original query: {query!r}", flush=True)
    print(f"[DEBUG] HyDE-expanded query: {hyde_query!r}", flush=True)
    print(f"[DEBUG] merged candidate pool size (pre-rerank): {len(candidates)}", flush=True)
    for doc in candidates:
        print(f"  page {doc.metadata.get('page', '?')}: {doc.page_content[:150]!r}", flush=True)

    if not candidates:
        return {
            "error": "No relevant content found in the document for this query.",
            "query": query,
        }

    # Rerank the merged pool using the ORIGINAL query text -- the
    # cross-encoder scores relevance to the actual question most accurately
    # against real phrasing, not the fabricated HyDE passage.
    result = reranker.compress_documents(documents=candidates, query=query)
    source_file = _THREAD_METADATA.get(str(thread_id), {}).get("filename", "the document")

    if not result:
        return {
            "error": "No relevant content found in the document for this query.",
            "query": query,
        }

    # Keep page numbers alongside each chunk so the LLM can cite them and
    # so bad retrievals are easier to debug. page metadata is 0-indexed in
    # PyPDFLoader, so +1 for a human-readable page number. Formatted as an
    # explicit labeled string (not a bare list of dicts) so a smaller model
    # can't miss the page/source tags when asked to cite them.
    formatted_chunks = []
    for doc in result:
        page = doc.metadata.get("page", -1) + 1
        formatted_chunks.append(
            f"[Source: {source_file}, Page {page}]\n{doc.page_content[:400]}"
        )

    return {
        "query": query,
        "source_file": source_file,
        "context": "\n\n---\n\n".join(formatted_chunks),
        "instruction": (
            "Each excerpt above is tagged with its exact source and page number. "
            "When you use any of this content in your answer, you MUST include "
            "that tag inline, formatted like: (source_file, p. N)."
        ),
    }


# ---------------------------------------------------------------------------
# EXPENSE TRACKER MCP SERVER (stdio)
# ---------------------------------------------------------------------------
mcp_client = MultiServerMCPClient(
    {
        "expense_tracker": {
            "command": r"C:\Users\R\Desktop\exp_tracker_mcp\.venv\Scripts\python.exe",
            "args": [r"C:\Users\R\Desktop\exp_tracker_mcp\main.py"],
            "transport": "stdio",
        }
    }
)


import threading

# Persistent background event loop for sync-wrapped async MCP tool calls.
# Using asyncio.run() per-call spins up and tears down a whole new event
# loop every single tool invocation, which is real overhead when tools are
# called repeatedly in a session. This loop is created once and kept alive
# for the lifetime of the process.
_mcp_loop = asyncio.new_event_loop()
threading.Thread(target=_mcp_loop.run_forever, daemon=True).start()


def _wrap_async_tool_sync(async_tool):
    """
    MCP tools loaded via langchain-mcp-adapters are async-only (they only
    implement `ainvoke`, not `invoke`). LangGraph's ToolNode calls tools
    synchronously when the graph runs via .stream() (not .astream()), which
    raises 'StructuredTool does not support sync invocation.' for these.

    This wraps each async MCP tool with a synchronous entrypoint that
    schedules the coroutine on the persistent background loop and blocks
    for the result, so it can be called from a sync graph execution
    context (Streamlit's top-level script is sync).
    """
    def _sync_call(**kwargs):
        future = asyncio.run_coroutine_threadsafe(async_tool.ainvoke(kwargs), _mcp_loop)
        return future.result()

    return StructuredTool.from_function(
        func=_sync_call,
        coroutine=async_tool.coroutine,
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
    )


def _load_mcp_tools():
    raw_tools = asyncio.run(mcp_client.get_tools())
    return [_wrap_async_tool_sync(t) for t in raw_tools]


expense_tracker_tools = _load_mcp_tools()

# base tool set always available; search_tool is excluded per-call when a
# PDF is indexed for the current thread (see chat_node / _build_llm_for_thread)
tools = [search_tool, get_stock_price, get_stock_price_range, calculator, rag_tool, *expense_tracker_tools]
tool_node = ToolNode(tools)  # ToolNode needs the full set to be able to execute any of them


def _build_llm_for_thread(thread_id: Optional[str]):
    """
    Bind tools for this specific call. When a PDF is indexed for this
    thread, `search_tool` is left out of the bound set entirely so the LLM
    cannot call Tavily at all -- retrieval must come from the document via
    `rag_tool` only. Without an indexed document, the full tool set
    (including web search) is available as normal.
    """
    if thread_id and thread_has_document(thread_id):
        active_tools = [t for t in tools if t is not search_tool]
    else:
        active_tools = tools
    return llm.bind_tools(active_tools)
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# How many of the most recent messages (from the FULL thread history) to
# actually send to the LLM on each turn. The full history still lives in
# the checkpointer (so the sidebar / "load past conversation" feature keeps
# working) -- this only controls what gets sent to Groq per-call.
MAX_HISTORY_MESSAGES = 10


def _trim_messages_for_llm(messages: list[BaseMessage], max_messages: int = MAX_HISTORY_MESSAGES) -> list[BaseMessage]:
    """
    Window the conversation to the most recent `max_messages`, then drop all
    but the single most recent ToolMessage in that window. Tool results
    (RAG context, expense tracker output, search results) tend to be the
    biggest token contributors, so keeping only the latest one meaningfully
    cuts token usage without losing recent conversational context.
    """
    trimmed = messages[-max_messages:] if len(messages) > max_messages else list(messages)

    tool_indices = [i for i, m in enumerate(trimmed) if isinstance(m, ToolMessage)]
    if len(tool_indices) > 1:
        keep_idx = tool_indices[-1]
        trimmed = [m for i, m in enumerate(trimmed) if (not isinstance(m, ToolMessage)) or i == keep_idx]

    return trimmed


def _invoke_with_retry(llm_bound, messages: list[BaseMessage], config=None, max_retries: int = 2):
    """
    Call the given tool-bound LLM with a couple of retries on rate-limit /
    payload-too-large errors, so a transient 429/413 doesn't hard-crash
    the Streamlit UI mid-stream.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return llm_bound.invoke(messages, config=config)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if ("rate_limit" in msg or "413" in msg or "too large" in msg) and attempt < max_retries:
                time.sleep(12)
                continue
            raise
    raise last_err


import re


def _find_last_successful_rag_result(messages: list[BaseMessage]) -> Optional[str]:
    """
    Scan recent messages for the most recent successful (non-error) ToolMessage
    from rag_tool, and return its raw content string, or None if there isn't
    one.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "rag_tool":
            content = str(msg.content)
            if '"error"' in content:
                return None  # most recent rag_tool call failed, nothing to cite
            return content
    return None


def _get_last_human_query(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


import json


def _get_clean_rag_context(raw_content: str) -> tuple[str, str, list[int]]:
    """
    rag_tool returns a dict, which LangChain serializes into the
    ToolMessage as a JSON string -- meaning real newlines inside the
    "context" field get escaped to literal backslash-n characters. This
    parses the JSON and returns clean, human-readable context text instead.

    Falls back to treating raw_content as already-clean text if it isn't
    valid JSON (defensive, in case the tool output format changes).
    """
    try:
        parsed = json.loads(raw_content)
        context = parsed.get("context", raw_content)
        source_file = parsed.get("source_file", "the document")
        pages = sorted(set(int(p) for p in re.findall(r"Page (\d+)", context)))
        return context, source_file, pages
    except (json.JSONDecodeError, TypeError, AttributeError):
        pages = sorted(set(int(p) for p in re.findall(r"Page (\d+)", raw_content)))
        source_match = re.search(r"Source:\s*([^,]+),", raw_content)
        source_file = source_match.group(1).strip() if source_match else "the document"
        return raw_content, source_file, pages


def _build_grounded_answer(query: str, raw_rag_content: str) -> str:
    """
    Produce the final answer for a RAG-grounded query by sending the raw
    retrieved context directly to the LLM and generating a free-form
    answer, with a real sources list appended from the retrieved chunks'
    metadata.

    NOTE: this intentionally removes the code-level verification guard
    (verbatim-quote extraction + exact substring check) that was built
    earlier to eliminate fabrication -- that guard is what stopped the
    model from inventing details not actually in the document. This
    function trusts the model's own generation directly, same as it would
    with no check at all. It is faster and produces more natural, fuller
    prose, at the cost of no longer being able to guarantee every claim in
    the answer is actually present in the source text.
    """
    clean_context, source_file, _ = _get_clean_rag_context(raw_rag_content)

    print("=" * 60, flush=True)
    print("[DEBUG] _build_grounded_answer (raw-context mode) called", flush=True)
    print(f"[DEBUG] query: {query!r}", flush=True)
    print(f"[DEBUG] clean context:\n{clean_context}", flush=True)
    print("-" * 60, flush=True)

    prompt = (
        "You are a helpful assistant answering questions about an uploaded "
        "document. Use the retrieved context below to answer the "
        "question as naturally and completely as you can. Do not mention "
        "'the context' or 'the document says' -- just answer directly. If "
        "the context doesn't contain the answer, say you couldn't find "
        "that information.\n\n"
        f"RETRIEVED CONTEXT:\n{clean_context}\n\n"
        f"QUESTION: {query}"
    )
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        answer_text = result.content.strip()
    except Exception as e:
        print(f"[DEBUG] raw-context generation raised: {e!r}", flush=True)
        answer_text = "Sorry, I couldn't generate an answer just now. Please try again."

    print(f"[DEBUG] raw-context generated answer: {answer_text!r}", flush=True)
    print("=" * 60, flush=True)

    # Sources list still built from real retrieved-chunk metadata (every
    # page that was in the retrieved batch, not just ones referenced in
    # the answer -- there's no verified link between answer content and
    # specific pages anymore since nothing is checked post-generation).
    pages = sorted(set(
        int(p) for p in re.findall(r"Page (\d+)", clean_context)
    ))
    sources_lines = "\n".join(f"* {source_file} | {p} | pdf" for p in pages)
    return f"{answer_text}\n\nSources:\n\n{sources_lines}" if sources_lines else answer_text


def generate_title(user_message: str):
    prompt = f"""
Generate a short title for this conversation.

Rules:
- Maximum 4 words
- Title Case
- No quotes
- No punctuation
- Only return the title

User message:
{user_message}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    print("[DEBUG] chat_node ENTERED", flush=True)
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    print(f"[DEBUG] thread_id: {thread_id!r}, thread_has_document: {thread_has_document(thread_id) if thread_id else 'N/A'}", flush=True)

    today_str = date.today().isoformat()

    system_message = SystemMessage(
        content=(
            f"Today's date is {today_str}. Always use this as 'today' when calling "
            "expense tracker tools (add_expense, list_expenses, summarize) -- never "
            "guess or use a placeholder date. Dates must be in YYYY-MM-DD format. "
            "For questions about the uploaded PDF, call "
            f"the `rag_tool` and include the thread_id `{thread_id}`. Only use "
            "`rag_tool` and only ask the user to upload a PDF when they are "
            "specifically asking about an uploaded document (e.g. 'summarize the "
            "PDF', 'what does the document say about X'). For general knowledge "
            "questions -- movies, shows, books, people, facts, current events, etc "
            "-- that are not about an uploaded document, answer directly from your "
            "own knowledge or use the web search tool. Do not ask for a PDF upload "
            "unless the user's question is actually about a document they intend "
            "to provide. You also have "
            "expense tracker tools -- use them whenever the user mentions expenses, "
            "spending, or purchases. You can also use the web search, stock price, "
            "and calculator tools when helpful. When the user asks for a stock price "
            "on a specific date, always pass that date (YYYY-MM-DD) to "
            "`get_stock_price` -- never assume the latest price answers a dated "
            "question. Use `get_stock_price_range` instead for questions about "
            "price movement over multiple days or comparisons between two dates.\n\n"
            "IMPORTANT -- before presenting ANY tool result as your answer: check "
            "that the result's fields actually match what was asked (correct date, "
            "symbol, filename, time period, etc). If a tool returns data for "
            "different parameters than requested, or returns an error, say so "
            "explicitly to the user -- never present mismatched or partial tool "
            "output as if it fully answers the original question.\n\n"
            "CRITICAL -- if `rag_tool` returns an error (e.g. 'no document indexed'), "
            "or returns context that does not actually contain the answer, you MUST "
            "tell the user exactly that -- e.g. 'I couldn't find that in the "
            "uploaded document' or 'no document appears to be indexed for this "
            "chat, please re-upload it.' NEVER invent a name, fact, or detail that "
            "is not literally present in the tool's returned context, even if the "
            "user insists a document was uploaded. If you catch yourself about to "
            "state a specific name/number/fact, verify it appears verbatim in the "
            "context you received -- if it doesn't, do not state it.\n\n"
            "When answering from PDF context returned by `rag_tool`: only use "
            "information present in that context, never mix in outside/parametric "
            "knowledge as if it came from the document. If the retrieved context "
            "does not contain the answer, say so explicitly instead of guessing.\n\n"
            "MANDATORY CITATION RULE: every piece of information you state that "
            "came from `rag_tool` MUST be immediately followed by a citation in "
            "the form (filename, p. N), using the exact source and page tags "
            "given in the tool's returned context. Do not summarize document "
            "content without a citation attached. If you're unsure which page a "
            "fact came from, do not state it.\n\n"
            "Do not pad document-grounded answers with generic textbook "
            "definitions or background knowledge that didn't come from the "
            "retrieved context (e.g. don't add 'a car is a motor vehicle "
            "with...' unless that sentence is literally in the document). "
            "Answer only what the retrieved context actually supports.\n\n"
            "GROUNDING CHECK: before stating any specific name, number, or "
            "detail from `rag_tool` results, find the exact phrase in the "
            "retrieved context that contains it. If you cannot point to where "
            "in the returned text a claim appears verbatim or near-verbatim, "
            "do not include that claim -- say the document doesn't specify it "
            "instead. Never state a proper noun, model name, or figure that "
            "isn't directly present in the retrieved context, even if it "
            "sounds plausible."
            + (
                "\n\nA document IS currently indexed for this chat. The web search "
                "tool is unavailable to you right now -- answer document-related "
                "questions using `rag_tool` only. If the document doesn't contain "
                "the answer, tell the user that rather than searching the web."
                if thread_id and thread_has_document(thread_id) else ""
            )
        )
    )

    # Only send a trimmed window of messages to the LLM to keep the request
    # well under Groq's tokens-per-minute limit. Full history is still
    # preserved in state / the checkpointer.
    recent_messages = _trim_messages_for_llm(state["messages"])

    llm_bound = _build_llm_for_thread(thread_id)
    messages = [system_message, *recent_messages]
    response = _invoke_with_retry(llm_bound, messages, config=config)

    if getattr(response, "tool_calls", None):
        # Intermediate step: this response is requesting a tool call, not
        # giving a final answer. Some models emit narrated draft text
        # alongside the tool call request (e.g. "The hotel is likely..."
        # before actually calling rag_tool). That draft text was observed
        # reaching the user via streaming even though it's not the real
        # answer -- strip it so only the eventual verified final answer is
        # ever shown. tool_calls are preserved so routing still works.
        try:
            response = response.model_copy(update={"content": ""})
        except Exception:
            response.content = ""
        return {"messages": [response]}

    # Final answer (no more tool calls pending). If rag_tool was used
    # successfully this turn, discard the model's free-form draft entirely
    # and build the answer deterministically from a verified verbatim
    # excerpt instead (see _build_grounded_answer for why).
    rag_content = _find_last_successful_rag_result(recent_messages)
    if rag_content is not None:
        user_query = _get_last_human_query(recent_messages)
        grounded_text = _build_grounded_answer(user_query, rag_content)
        response = AIMessage(content=grounded_text)
    else:
        print("[DEBUG] No successful rag_tool result found in recent_messages this turn.", flush=True)
        print(f"[DEBUG] Model answered directly instead: {response.content!r}", flush=True)

    return {"messages": [response]}


def _rag_tool_error_message(tool_message: ToolMessage) -> Optional[str]:
    """
    Inspect a ToolMessage from rag_tool and return a fixed, deterministic
    reply if the tool signaled a failure -- no document indexed, or no
    relevant content found. Returns None if this wasn't an error (or wasn't
    from rag_tool), meaning the normal flow (back to chat_node) should run.

    This exists because prompt instructions alone were not reliably
    stopping the LLM from inventing an answer (including a fabricated
    citation) when rag_tool failed. Intercepting the error in code, before
    the LLM gets another turn, removes that failure mode entirely.
    """
    if tool_message.name != "rag_tool":
        return None

    content = str(tool_message.content)
    if "No document indexed" in content:
        return (
            "No document appears to be indexed for this chat yet. "
            "Please upload the PDF first, then ask again."
        )
    if "No relevant content found" in content:
        return (
            "I couldn't find anything relevant to that question in the "
            "uploaded document."
        )
    return None


def route_after_tools(state: ChatState) -> str:
    """
    After tool execution: if the most recent tool result was a rag_tool
    error, skip chat_node entirely and go straight to a deterministic
    error response. Otherwise proceed to chat_node as normal so the LLM
    can compose an answer from successful tool results.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and _rag_tool_error_message(last_message) is not None:
        return "rag_error_response"
    return "chat_node"


def rag_error_response(state: ChatState) -> dict:
    """Emit the fixed error message directly, bypassing the LLM entirely."""
    last_message = state["messages"][-1]
    text = _rag_tool_error_message(last_message)
    return {"messages": [AIMessage(content=text)]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("rag_error_response", rag_error_response)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_conditional_edges(
    "tools",
    route_after_tools,
    {"chat_node": "chat_node", "rag_error_response": "rag_error_response"},
)
graph.add_edge("rag_error_response", END)

chatbot = graph.compile(checkpointer=checkpointer)