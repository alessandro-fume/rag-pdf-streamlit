import os
import re
import time
import hashlib
import pathlib
import shutil
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

# =============================
# Configurazione base
# =============================
APP_TITLE = "Chatbot EOS Reply – Gestione Documenti (Web Ready)"
APP_ICON = "📚"
BASE_DIR = "vectorstore"  # cartella PERSISTENTE tra esecuzioni per gli indici

# Marker di fiducia per caricare indici in sicurezza
TRUST_MARK_FILE = ".trusted_by_app"
TRUST_MARK_VALUE = "EOS-REPLY-RAG-V1"
LEGACY_PREFIXES = ("ws_",)  # vecchie cartelle create prima di questa versione

# Template semplici inline per evitare dipendenze esterne
CSS = """
<style>
    .chat-msg { padding: 0.75rem 1rem; border-radius: 12px; margin: 0.25rem 0; }
    .user { background: #eef6ff; border: 1px solid #cfe4ff; }
    .bot { background: #f7f7ff; border: 1px solid #e5e5ff; }
    .msg { white-space: pre-wrap; }
    .muted { color: #6b7280; font-size: 0.9rem; }
</style>
"""

USER_TEMPLATE = """
<div class="chat-msg user"><div class="msg">{msg}</div></div>
"""
BOT_TEMPLATE = """
<div class="chat-msg bot"><div class="msg">{msg}</div></div>
"""

# =============================
# Utility
# =============================

def get_openai_key_from_secrets() -> str:
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("❌ Nessuna chiave API trovata. Imposta st.secrets['OPENAI_API_KEY'].")
        st.stop()
    os.environ["OPENAI_API_KEY"] = key  # richiesto da OpenAIEmbeddings / ChatOpenAI
    return key


def ensure_store_dir() -> str:
    """Ritorna la cartella *globale e persistente* che contiene tutti gli indici."""
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
            if d.is_dir():
                # Mostra indici "trusted" o legacy (ws_*)
                if _has_trust_mark(d.path) or any(d.name.startswith(p) for p in LEGACY_PREFIXES):
                    items.append(d.name)
    # Ordina alfabeticamente per comodità
    return sorted(items)


# =============================
# PDF → Testo
# =============================

def extract_text_from_pdfs(pdfs) -> str:
    """Estrae tutto il testo (nessun limite di pagine/size). Se il PDF è scannerizzato senza testo, il risultato può essere vuoto."""
    text = ""
    for pdf in pdfs:
        try:
            with pdfplumber.open(pdf) as reader:
                for page in reader.pages:
                    content = page.extract_text() or ""
                    text += content
        except Exception as e:
            st.warning(f"⚠️ Errore lettura PDF: {e}")
    return text


def chunk_text(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,    # chunk più grandi per documenti estesi
        chunk_overlap=100,  # piccolo overlap
        length_function=len,
    )
    return splitter.split_text(text)


# =============================
# Vettori & Chain
# =============================

def build_vectorstore(chunks, path: str):
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_texts(texts=chunks, embedding=embeddings)
    os.makedirs(path, exist_ok=True)
    vs.save_local(path)
    # Scrivi il marker di fiducia per riconoscere che l'indice è stato creato da questa app
    try:
        with open(os.path.join(path, TRUST_MARK_FILE), "w", encoding="utf-8") as f:
            f.write(TRUST_MARK_VALUE)
    except Exception:
        pass
    return vs


@st.cache_resource(show_spinner=False)
def load_vectorstore(path: str):
    name = os.path.basename(path)
    # Consenti il caricamento SOLO se la cartella è marcata come nostra o è legacy (ws_*)
    if not (_has_trust_mark(path) or any(name.startswith(p) for p in LEGACY_PREFIXES)):
        st.error("Per sicurezza, l'indice selezionato non è riconosciuto come creato da questa app.")
        st.stop()
    # Abilita la deserializzazione "pericolosa" SOLO su indici che arrivano da questa app
    return FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)


def build_chain(vectorstore, temperature: float):
    llm = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo")
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )
    return chain


# =============================
# App
# =============================

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON)
    get_openai_key_from_secrets()

    st.markdown(CSS, unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption("RAG su PDF – versione pronta per deploy web. Nessun limite di pagine/size (attenzione a tempi/costi).")

    # Stato sessione
    defaults = {
        "conversation": None,
        "chat_history": None,
        "last_text_preview": "",
        "current_index": None,
        "confirm_delete": False,
        "llm_temperature": 0.3,
        "messages": [],         # cronologia persistente lato UI
        "last_sources": [],     # fonti dell'ultima risposta
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.subheader("⚙️ Impostazioni")
        st.session_state.llm_temperature = st.slider("Temperatura del modello", 0.0, 1.0, 0.3, step=0.1)

        st.subheader("📁 Indici disponibili (persistenti)")
        indices = list_indices()
        selected = st.selectbox("Scegli un documento indicizzato", options=["-- Nessuno --"] + indices)

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

        st.markdown("---")
        st.subheader("📤 Carica nuovo PDF da indicizzare")
        pdf_doc = st.file_uploader("Carica un PDF", type=["pdf"], accept_multiple_files=False)
        overwrite = st.checkbox("Sovrascrivi se già esistente", value=False)

        if pdf_doc and st.button("Processa e indicizza"):
            with st.spinner("📚 Indicizzazione in corso (potrebbe richiedere tempo per PDF molto grandi)..."):
                try:
                    filename = safe_name(pdf_doc.name)
                    base = ensure_store_dir()
                    vectorstore_path = os.path.join(base, filename)

                    if os.path.exists(vectorstore_path) and not overwrite:
                        st.warning("Documento già processato. Attiva la sovrascrittura per ri-elaborarlo.")
                        st.stop()

                    raw_text = extract_text_from_pdfs([pdf_doc])
                    if not raw_text.strip():
                        st.error("❌ Nessun testo leggibile trovato nel PDF (forse è scannerizzato). Aggiungi OCR se necessario.")
                        st.stop()

                    chunks = chunk_text(raw_text)
                    vs = build_vectorstore(chunks, vectorstore_path)

                    # Salva una copia del PDF accanto all'indice (per anteprima)
                    try:
                        with open(os.path.join(vectorstore_path, "documento.pdf"), "wb") as f:
                            f.write(pdf_doc.getbuffer())
                    except Exception:
                        pass

                    st.session_state.last_text_preview = raw_text[:3000]
                    st.session_state.conversation = build_chain(vs, st.session_state.llm_temperature)
                    st.session_state.current_index = filename

                    # reset cronologia per la nuova sessione su questo indice
                    st.session_state.messages = []
                    st.session_state.last_sources = []

                    st.success(f"✅ Documento '{filename}' indicizzato e pronto all'uso!")
                except Exception as e:
                    st.error(f"❌ Errore durante l'elaborazione: {e}")

    # Caricamento indice esistente (selezionato da sidebar)
    if selected != "-- Nessuno --" and selected != st.session_state.current_index:
        with st.spinner("🔁 Caricamento indice..."):
            try:
                vectorstore_path = os.path.join(ensure_store_dir(), selected)
                vs = load_vectorstore(vectorstore_path)
                st.session_state.conversation = build_chain(vs, st.session_state.llm_temperature)
                st.session_state.current_index = selected
                st.success(f"✅ Indice '{selected}' caricato!")

                pdf_path = os.path.join(vectorstore_path, "documento.pdf")
                if os.path.exists(pdf_path):
                    try:
                        with pdfplumber.open(pdf_path) as pdf_file:
                            text = "".join([p.extract_text() or "" for p in pdf_file.pages])
                            st.session_state.last_text_preview = text[:3000]
                    except Exception:
                        st.session_state.last_text_preview = ""

                # reset della chat quando cambi indice
                st.session_state.messages = []
                st.session_state.last_sources = []

            except Exception as e:
                st.error(f"❌ Errore nel caricamento dell'indice: {e}")

    # Anteprima testo
    if st.session_state.last_text_preview:
        with st.expander("📄 Anteprima testo estratto"):
            st.text_area("Testo", value=st.session_state.last_text_preview, height=240)

    # Suggerimenti
    if st.session_state.conversation:
        with st.expander("💡 Suggerimenti di domande"):
            st.markdown("- **Qual è l’argomento principale del documento?**")
            st.markdown("- **Fammi un riassunto per avere il contesto.**")
            st.markdown("- **Quali sono i punti chiave trattati?**")

    # Prompt utente
    user_question = st.text_input("Fai una domanda sul documento selezionato:")

    if user_question:
        if not st.session_state.conversation:
            st.warning("⚠️ Seleziona o carica prima un documento.")
        else:
            with st.spinner("🧠 Elaborazione..."):
                try:
                    # 1) salva la domanda dell'utente
                    st.session_state.messages.append({"role": "user", "content": user_question})

                    # 2) esegui la chain
                    response = st.session_state.conversation({"question": user_question})
                    answer = response["answer"]

                    # 3) salva la risposta del bot e le fonti usate
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_sources = response.get("source_documents", [])

                except Exception as e:
                    st.error(f"❌ Errore durante la risposta: {e}")

    # --- RENDER COMPLETO DELLA CHAT (persistente) ---
    if st.session_state.get("messages"):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(USER_TEMPLATE.format(msg=msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(BOT_TEMPLATE.format(msg=msg["content"]), unsafe_allow_html=True)

        # Mostra le fonti dell'ULTIMA risposta
        if st.session_state.get("last_sources"):
            with st.expander("🔍 Contenuti utilizzati per l'ultima risposta"):
                for i, doc in enumerate(st.session_state.last_sources):
                    st.markdown(f"**Chunk {i+1}:**

{doc.page_content}

---")

    # Informativa rapida
    with st.expander("ℹ️ Note importanti"):
        st.markdown(
            """
- **Nessun limite di pagine/size**: l'elaborazione di PDF molto grandi può richiedere tempo e generare costi API più alti.
- **PDF scannerizzati** senza layer di testo potrebbero risultare vuoti: per tali file serve un OCR (opzionale, non incluso in questa versione).
- **Indici PERSISTENTI**: la lista a sinistra mostra gli indici creati da questa app (o legacy `ws_*`).
- **Sicurezza**: il caricamento di indici abilita la deserializzazione solo per cartelle marcate come "trusted" o legacy note.
- **Persistenza Cloud**: su Streamlit Cloud lo storage può essere effimero; per persistenza reale usa uno storage esterno (S3/MinIO/GCS).
            """
        )


if __name__ == "__main__":
    main()

