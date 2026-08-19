import streamlit as st
from backend_mcp import chatbot, generate_title, ingest_pdf, thread_document_metadata
from langchain_core.messages import HumanMessage, AIMessage
import uuid

APP_NAME = "RAG Chatbot"
APP_TAGLINE = "Ask questions about your uploaded PDF."

# *************************** PAGE CONFIG & STYLING ***************************

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ---- global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* ---- app header ---- */
    .verity-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.1rem;
    }
    .verity-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7C5CFC 0%, #4FD1C5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .verity-tagline {
        color: #8A90A6;
        font-size: 0.95rem;
        margin-bottom: 1.4rem;
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #10131C;
        border-right: 1px solid #232838;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #E6E8F0 !important;
        font-weight: 700;
    }

    /* ---- document status card ---- */
    .doc-card {
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.88rem;
        border: 1px solid #232838;
    }
    .doc-card.active {
        background: linear-gradient(135deg, rgba(124,92,252,0.15), rgba(79,209,197,0.10));
        border: 1px solid rgba(124,92,252,0.4);
        color: #C9CCE6;
    }
    .doc-card.empty {
        background: #151A24;
        color: #7A8099;
    }

    /* ---- conversation list buttons ---- */
    section[data-testid="stSidebar"] .stButton button {
        background-color: transparent;
        border: 1px solid #232838;
        color: #C9CCE6;
        text-align: left;
        border-radius: 8px;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        border-color: #7C5CFC;
        color: #FFFFFF;
        background-color: rgba(124,92,252,0.10);
    }

    /* ---- chat bubbles ---- */
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.4rem 0.2rem;
        margin-bottom: 0.4rem;
    }

    /* ---- chat input ---- */
    .stChatInput textarea {
        border-radius: 12px !important;
        border: 1px solid #2A3040 !important;
    }

    /* ---- sources styling inside answers ---- */
    .stChatMessage p {
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="verity-header">
        <span style="font-size:2rem;">🔎</span>
        <span class="verity-title">{APP_NAME}</span>
    </div>
    <div class="verity-tagline">{APP_TAGLINE}</div>
    """,
    unsafe_allow_html=True,
)

# *************************** UTILITY FUNCTIONS ***************************

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = "New Chat" ### initially every chat is termed as new chat .
                                                                 ### chat_threads does contain id as well as new chat topic therfore it is a dict now.

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])  #get state helps to retrieve all messages for one particular thread id .


# *************************** SESSION SETUP ***************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Dictionary
# {
#   thread_id : conversation_title
# }
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = {}

add_thread(st.session_state["thread_id"])

current_thread_id = st.session_state["thread_id"]

# *************************** SIDEBAR ***************************

with st.sidebar:
    st.markdown("## 🔎 " + APP_NAME)

    if st.button("＋ New Chat", use_container_width=True):
        reset_chat()

    st.markdown("### 📄 Document")

    doc_meta = thread_document_metadata(current_thread_id)
    if doc_meta:
        st.markdown(
            f"""<div class="doc-card active">
                ✅ <b>{doc_meta.get('filename')}</b><br>
                {doc_meta.get('chunks')} chunks · {doc_meta.get('documents')} pages
                </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="doc-card empty">No PDF indexed for this chat yet.</div>""",
            unsafe_allow_html=True,
        )

    # NOTE: the uploader is keyed to current_thread_id. Without this, Streamlit
    # persists the widget's uploaded file across reruns -- including after
    # switching to a *different* thread via the sidebar -- which would silently
    # re-ingest a previous thread's PDF into a new, unrelated thread the next
    # time this block runs. Scoping the key to the thread makes the widget
    # reset to empty whenever the active thread changes.
    uploaded_pdf = st.file_uploader(
        "Upload a PDF for this chat",
        type=["pdf"],
        key=f"pdf_uploader_{current_thread_id}",
        label_visibility="collapsed",
    )

    if uploaded_pdf is not None and doc_meta.get("filename") != uploaded_pdf.name:
        with st.status(f"Indexing '{uploaded_pdf.name}'...", expanded=False) as status:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=current_thread_id,
                    filename=uploaded_pdf.name,
                )
                status.update(
                    label=f"Indexed '{summary['filename']}' ({summary['chunks']} chunks).",
                    state="complete",
                )
            except Exception as e:
                status.update(label=f"Failed to index PDF: {e}", state="error")

    st.markdown("### 💬 Conversations")

    for thread_id, title in reversed(list(st.session_state["chat_threads"].items())):

        is_active = thread_id == current_thread_id
        label = f"{'🟣 ' if is_active else ''}{title}"

        if st.button(label, key=thread_id, use_container_width=True):

            st.session_state["thread_id"] = thread_id

            messages = load_conversation(thread_id)

            temp_messages = []   #doing thi below due to compatibility issue in load convo func get state return messages which includes metadata as well therefore to make it compatible below code is executed .

            for msg in messages:

                if isinstance(msg, HumanMessage):
                    role = "user"
                else:
                    role = "assistant"

                temp_messages.append(
                    {
                        "role": role,
                        "content": msg.content,
                    }
                )

            st.session_state["message_history"] = temp_messages
            st.rerun()


# *************************** MAIN UI ***************************

for message in st.session_state["message_history"]:

    avatar = "🧑" if message["role"] == "user" else "🔎"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])


user_input = st.chat_input("Ask something about your document…")


if user_input:

    # Save user message
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Generate title only once
    current_thread = st.session_state["thread_id"]

    if st.session_state["chat_threads"][current_thread] == "New Chat":

        title = generate_title(user_input)

        st.session_state["chat_threads"][current_thread] = title

    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {      ## langsmith concept of metadata is used here to store the run name for each chat turn which can be used to track the conversation in langsmith dashboard.
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn" , 
    }

    with st.chat_message("assistant", avatar="🔎"):
        with st.spinner("Thinking..."):
            # NOTE: intentionally NOT using stream_mode="messages" here.
            # chat_node makes multiple internal LLM calls per turn (HyDE
            # query expansion, the tool-calling decision, and answer
            # generation for grounded RAG answers). "messages" mode
            # streams raw live tokens from ALL of these as they're
            # generated -- including ones never meant to be user-visible,
            # like HyDE's fabricated hypothetical passage -- because
            # LangGraph streams provider-level tokens as they're produced,
            # before any Python-side post-processing/overriding in
            # chat_node has a chance to run. Using invoke() and displaying
            # only the final resolved state avoids this leak entirely.
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
            )
            final_messages = result.get("messages", [])
            ai_message = ""
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_message = msg.content
                    break

        st.write(ai_message)

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
        }
    )