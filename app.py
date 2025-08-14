import os
import re
import json
import pathlib
import shutil
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# =============================
# Branding / impostazioni base
# =============================
APP_TITLE = "Chatbot EOS Reply – PDF Documents"
FAVICON_PATH = "logo_favicon.png"   # quadrata, ottimizzata per tab browser
LOGO_PATH = "logo_eos_reply.png"    # logo in pagina/sidebar (orizzontale)
APP_ICON_FALLBACK = "📚"

BASE_DIR = "vectorstore"
TRUST_MARK_FILE = ".trusted_by_app"
TRUST_MARK_VALUE = "EOS-REPLY-RAG-V1"
LEGACY_PREFIXES = ("ws_",)

# =============================
# Stili e template
# =============================
CSS = """
<style>
    .chat-msg { padding: 0.75rem 1rem; border-radius: 12px; margin: 0.25rem 0; }
    .user { background: #eef6ff; border: 1px solid #cfe4ff; }
    .bot  { background: #f7f7ff; border: 1px solid #e5e5ff; }
    .msg  { white-space: pre-wrap; }
    .muted{ color: #6b7280; font-size: 0.9rem; }
    h1 { font-size: 2.6rem; }
</style>
"""
USER_TEMPLATE = """<div class="chat-msg user"><div class="msg">{msg}</div></div>"""
BOT_TEMPLATE  = """<div class="chat-msg bot"><div class="msg">{msg}</div></div>"""

# =============================
# Prompt QA anti-hallucinations
# =============================
QA_PROMPT = PromptTemplate.from_template("""
Sei un assistente tecnico. Rispondi SOLO usando il contenuto dei documenti forniti (contesto).
Regole:
- Se la domanda richiede una procedura, fornisci TUTTI i passaggi (in elenco numerato).
- Mantieni la risposta focalizzata, concreta e aderente al testo.
- Se la risposta è lunga, l'utente può scrivere "continua" per proseguire.
- Se l'informazione non è presente nei documenti o non sei sicuro, dillo chiaramente.
- Cita SEMPRE le pagine utilizzate alla fine (es: "Riferimenti: p. 12, 13").

Domanda: {question}

Contesto:
{context}

Risposta (markdown, includi elenco completo se applicabile):
""")

# =============================
# Utility
# =============================
def get_openai_key_from_secrets() -> str:
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("❌ Nessuna chiave API trovata. Imposta st.secrets['OPENAI_API_KEY'].")
        st.stop()
    os.environ["OPENAI_API_KEY"] = key
    return key

def ensure_store_dir() -> str:
    os.makedirs(BASE_DIR, exist_ok=True)
    return BASE_DIR

def safe_name(name: str) -> str:
    stem = pathlib.Path(name).stem
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)
    return stem[:64] or "doc"

def _has_trust_mark(dir_path: str) -> bool:
    mark_path = os.path.join(dir_path, TRUST_MARK_FILE)
    try:
        if os.path.exists(mark_path):
            with open(mark_path, "r", encoding="utf-8") as f:
                return f.read().strip() == TRUST_MARK_VALUE
    except Exception:
        return False
    return False

def list_indices() -> list:
    base = ensure_store_dir()
    items = []
    if os.path.exists(base):
        for d in os.scandir(base):
            if d.is_dir() and (_has_trust_mark(d.path) or any(d.name.startswith(p) for p in LEGACY_PREFIXES)):
                items.append(d.name)
    return sorted(items)

# =============================
# PDF → Documents con metadati pagina
# =============================
def documents_from_pdf_with_pages(uploaded_pdf) -> list[Document]:
    """
    Estrae il testo pagina per pagina e crea Document con metadati:
    - page: numero pagina (1-based)
    - source: nome file
    """
    docs = []
    file_name = getattr(uploaded_pdf, "name", "documento.pdf")
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    try:
        with pdfplumber.open(uploaded_pdf) as reader:
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    # se pagina vuota (scansione immagine senza OCR), salta
                    continue
                for chunk in splitter.split_text(text):
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={"page": i, "source": file_name}
                        )
                    )
    except Exception as e:
        st.warning(f"⚠️ Errore lettura PDF: {e}")
    return docs

# =============================
# Vector store & chain
# =============================
def build_vectorstore_from_documents(docs, path: str):
    # embeddings espliciti per coerenza tra salvataggio e caricamento
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vs = FAISS.from_documents(docs, embedding=embeddings)
    os.makedirs(path, exist_ok=True)
    vs.save_local(path)
    # marker di fiducia
    try:
        with open(os.path.join(path, TRUST_MARK_FILE), "w", encoding="utf-8") as f:
            f.write(TRUST_MARK_VALUE)
    except Exception:
        pass
    return vs

@st.cache_resource(show_spinner=False)
def load_vectorstore(path: str):
    name = os.path.basename(path)
    if not (_has_trust_mark(path) or any(name.startswith(p) for p in LEGACY_PREFIXES)):
        st.error("Indice non riconosciuto come creato da questa app.")
        st.stop()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def build_chain(vectorstore, temperature: float):
    # LLM con più spazio di output per elenchi completi
    llm = ChatOpenAI(
        temperature=0.2,                # più deterministico
        model_name="gpt-3.5-turbo",
        max_tokens=900                  # più margine per step lunghi
    )
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    # retriever più “ampio” per aumentare copertura
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        output_key="answer",
    )

# =============================
# Header con logo + titolo
# =============================
def render_header():
    """Mostra il logo a sinistra e il titolo a destra."""
    if os.path.exists(LOGO_PATH):
        col1, col2 = st.columns([0.12, 0.88])
        with col1:
            st.image(LOGO_PATH, width=160)
        with col2:
            st.markdown(
                f"<h1 style='margin-top:0.2rem; margin-bottom:0.2rem;'>{APP_TITLE}</h1>",
                unsafe_allow_html=True,
            )
    else:
        st.title(APP_TITLE)

# =============================
# App
# =============================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=FAVICON_PATH)
    get_openai_key_from_secrets()
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()

    # Stato
    defaults = {
        "conversation": None,
        "chat_history": None,
        "last_text_preview": "",
        "current_index": None,
        "confirm_delete": False,
        "llm_temperature": 0.3,
        "messages": [],
        "last_sources": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ---------- Sidebar ----------
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=120)
        st.subheader("⚙️ Impostazioni")
        st.session_state.llm_temperature = st.slider("Temperatura del modello", 0.0, 1.0, 0.5, step=0.1)

        st.subheader("📁 Indici disponibili (persistenti)")
        indices = list_indices()
        selected = st.selectbox("Scegli un documento indicizzato", options=["-- Nessuno --"] + indices)

        # Elimina indice
        if selected != "-- Nessuno --":
            col1, col2 = st.columns(2)
            with col1:
                if not st.session_state.confirm_delete and st.button("🗑️ Elimina indice"):
                    st.session_state.confirm_delete = True
                    st.warning("Premi di nuovo per confermare l'eliminazione.")
            with col2:
                if st.session_state.confirm_delete and st.button("✅ Conferma"):
                    try:
                        shutil.rmtree(os.path.join(ensure_store_dir(), selected))
                        st.success(f"Indice '{selected}' eliminato.")
                        # Reset totale stato e input
                        st.session_state.update({
                            "conversation": None,
                            "last_text_preview": "",
                            "current_index": None,
                            "confirm_delete": False,
                            "messages": [],
                            "last_sources": [],
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Errore durante l'eliminazione: {e}")
                        st.session_state.confirm_delete = False

        # Upload e indicizzazione
        st.markdown("---")
        st.subheader("📤 Carica nuovo PDF da indicizzare")
        pdf_doc = st.file_uploader("Carica un PDF", type=["pdf"], accept_multiple_files=False)
        overwrite = st.checkbox("Sovrascrivi se già esistente", value=False)

        if pdf_doc and st.button("Processa e indicizza"):
            with st.spinner("📚 Indicizzazione in corso..."):
                try:
                    filename = safe_name(pdf_doc.name)
                    vectorstore_path = os.path.join(ensure_store_dir(), filename)
                    if os.path.exists(vectorstore_path) and not overwrite:
                        st.warning("Documento già processato.")
                        st.stop()

                    # 1) Estrai Document con metadati pagina
                    docs = documents_from_pdf_with_pages(pdf_doc)
                    if not docs:
                        st.error("❌ Nessun testo leggibile trovato nel PDF (forse scannerizzato senza OCR).")
                        st.stop()

                    # 2) Crea vector store
                    vs = build_vectorstore_from_documents(docs, vectorstore_path)

                    # 3) Salva copia PDF per anteprima
                    try:
                        with open(os.path.join(vectorstore_path, "documento.pdf"), "wb") as f:
                            f.write(pdf_doc.getbuffer())
                    except Exception:
                        pass

                    # 4) Aggiorna stato + reset completo sessione (punto 2)
                    preview_text = "\n".join([d.page_content for d in docs[:4]])  # piccola anteprima
                    st.session_state.last_text_preview = preview_text[:3000]
                    st.session_state.conversation = build_chain(vs, st.session_state.llm_temperature)
                    st.session_state.current_index = filename
                    st.session_state.messages = []
                    st.session_state.last_sources = []
                    # pulizia input
                    st.session_state.pop("user_q", None)

                    st.success(f"✅ Documento '{filename}' indicizzato e pronto all'uso!")
                except Exception as e:
                    st.error(f"❌ Errore durante l'elaborazione: {e}")

    # ---------- Caricamento indice esistente ----------
    if selected != "-- Nessuno --" and selected != st.session_state.current_index:
        with st.spinner("🔁 Caricamento indice..."):
            try:
                vs = load_vectorstore(os.path.join(ensure_store_dir(), selected))
                st.session_state.conversation = build_chain(vs, st.session_state.llm_temperature)
                st.session_state.current_index = selected

                # Carica anteprima testo se disponibile
                pdf_path = os.path.join(ensure_store_dir(), selected, "documento.pdf")
                if os.path.exists(pdf_path):
                    try:
                        with pdfplumber.open(pdf_path) as pdf_file:
                            text = "".join([p.extract_text() or "" for p in pdf_file.pages])
                            st.session_state.last_text_preview = text[:3000]
                    except Exception:
                        st.session_state.last_text_preview = ""

                # Reset completo sessione e input (punto 2)
                st.session_state.messages = []
                st.session_state.last_sources = []
                st.session_state.pop("user_q", None)

                st.success(f"✅ Indice '{selected}' caricato!")
            except Exception as e:
                st.error(f"❌ Errore nel caricamento dell'indice: {e}")

    # ---------- Anteprima/Suggerimenti ----------
    if st.session_state.last_text_preview:
        with st.expander("📄 Anteprima testo estratto"):
            st.text_area("Testo", value=st.session_state.last_text_preview, height=240)

    if st.session_state.conversation:
        with st.expander("💡 Suggerimenti di domande"):
            st.markdown("- **Qual è l’argomento principale del documento?**")
            st.markdown("- **Fammi un riassunto per avere il contesto.**")
            st.markdown("- **Quali sono i punti chiave trattati?**")

    # ---------- Input domanda (FORM: si svuota automaticamente) ----------
    with st.form("qa_form", clear_on_submit=True):
        user_question = st.text_input("Fai una domanda sul documento selezionato:", key="user_q")
        submitted = st.form_submit_button("Invia")

    if submitted and user_question and user_question.strip():
        if not st.session_state.conversation:
            st.warning("⚠️ Seleziona o carica prima un documento.")
        else:
            with st.spinner("🧠 Elaborazione..."):
                try:
                    q = user_question.strip()
                    # 1) salva la domanda
                    st.session_state.messages.append({"role": "user", "content": q})
                    # 2) esegui la chain
                    response = st.session_state.conversation({"question": q})
                    answer = response["answer"]
                    # 3) salva la risposta e le fonti
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_sources = response.get("source_documents", [])
                except Exception as e:
                    st.error(f"❌ Errore durante la risposta: {e}")

    # ---------- Render chat ----------
    if st.session_state.get("messages"):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(USER_TEMPLATE.format(msg=msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(BOT_TEMPLATE.format(msg=msg["content"]), unsafe_allow_html=True)

        # Fonti: mostra pagina se presente nei metadati
        if st.session_state.get("last_sources"):
            with st.expander("🔍 Contenuti utilizzati per l'ultima risposta"):
                for i, doc in enumerate(st.session_state.last_sources, start=1):
                    page = doc.metadata.get("page")
                    src = doc.metadata.get("source", "")
                    header = f"**Chunk {i}**"
                    if page:
                        header += f" · p. {page}"
                    if src:
                        header += f" · {src}"
                    st.markdown(f"{header}\n\n{doc.page_content}\n\n---")

    # ---------- Note semplificate ----------
    with st.expander("ℹ️ Note importanti"):
        st.markdown(
            """
- Puoi caricare PDF di qualsiasi dimensione (i file molto grandi potrebbero richiedere più tempo).
- I PDF scannerizzati (solo immagine) potrebbero non contenere testo ricercabile.
- I documenti indicizzati restano disponibili finché l'app rimane attiva.
            """
        )

if __name__ == "__main__":
    main()
