import os
import re
import time
import hashlib
import pathlib
import shutil
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# =============================
# Configurazione base
# =============================
APP_TITLE = "Chatbot EOS Reply – Gestione Documenti (Web Ready)"
APP_ICON = "📚"
BASE_DIR = "vectorstore"  # verrà namespacizzata per sessione

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


def session_workspace() -> str:
    """Crea una cartella di lavoro isolata per la sessione utente (evita collisioni fra utenti)."""
    if "workspace_dir" not in st.session_state:
        suffix = hashlib.sha1(f"{time.time()}-{st.session_state.get('_session_id','')}".encode()).hexdigest()[:8]
        ws = os.path.join(BASE_DIR, f"ws_{suffix}")
        os.makedirs(ws, exist_ok=True)
        st.session_state.workspace_dir = ws
    return st.session_state.workspace_dir


def safe_name(name: str) -> str:
    stem = pathlib.Path(name).stem
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)
    return stem[:64] or "doc"


def list_indices() -> list:
    base = session_workspace()
    return [d.name for d in os.scandir(base) if d.is_dir()] if os.path.exists(base) else []


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
    return vs


@st.cache_resource(show_spinner=False)
def load_vectorstore(path: str):
    return FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=False)


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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.subheader("⚙️ Impostazioni")
        st.session_state.llm_temperature = st.slider("Temperatura del modello", 0.0, 1.0, 0.3, step=0.1)

        st.subheader("📁 Indici nella tua workspace")
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
                        shutil.rmtree(os.path.join(session_workspace(), selected))
                        st.success(f"Indice '{selected}' eliminato.")
                        st.session_state.update({
                            "conversation": None,
                            "last_text_preview": "",
                            "current_index": None,
                            "confirm_delete": False,
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
                    vectorstore_path = os.path.join(session_workspace(), filename)

                    if os.path.exists(vectorstore_path) and not overwrite:
                        st.warning("Documento già processato in questa workspace. Attiva la sovrascrittura per ri-elaborarlo.")
                        st.stop()

                    raw_text = extract_text_from_pdfs([pdf_doc])
                    if not raw_text.strip():
                        st.error("❌ Nessun testo leggibile trovato nel PDF (forse è scannerizzato). Aggiungi OCR se necessario.")
                        st.stop()

                    chunks = chunk_text(raw_text)
                    vs = build_vectorstore(chunks, vectorstore_path)

                    # Salva una copia del PDF accanto all'indice (comodo per anteprima)
                    with open(os.path.join(vectorstore_path, "documento.pdf"), "wb") as f:
                        f.write(pdf_doc.getbuffer())

                    st.session_state.last_text_preview = raw_text[:3000]
                    st.session_state.conversation = build_chain(vs, st.session_state.llm_temperature)
                    st.session_state.current_index = filename

                    st.success(f"✅ Documento '{filename}' indicizzato e pronto all'uso!")
                except Exception as e:
                    st.error(f"❌ Errore durante l'elaborazione: {e}")

    # Caricamento indice esistente (selezionato da sidebar)
    if selected != "-- Nessuno --" and selected != st.session_state.current_index:
        with st.spinner("🔁 Caricamento indice..."):
            try:
                vectorstore_path = os.path.join(session_workspace(), selected)
                vs = load_vectorstore(vectorstore_path)
                st.session_state.conversation = build_chain(vs, st.session_state.llm_temperature)
                st.session_state.current_index = selected
                st.success(f"✅ Indice '{selected}' caricato!")

                pdf_path = os.path.join(vectorstore_path, "documento.pdf")
                if os.path.exists(pdf_path):
                    with pdfplumber.open(pdf_path) as pdf_file:
                        text = "".join([p.extract_text() or "" for p in pdf_file.pages])
                        st.session_state.last_text_preview = text[:3000]
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
                    response = st.session_state.conversation({"question": user_question})
                    st.session_state.chat_history = response["chat_history"]

                    # Chat history
                    for i, msg in enumerate(st.session_state.chat_history):
                        html = USER_TEMPLATE.format(msg=msg.content) if i % 2 == 0 else BOT_TEMPLATE.format(msg=msg.content)
                        st.markdown(html, unsafe_allow_html=True)

                    # Sorgenti
                    if "source_documents" in response and response["source_documents"]:
                        with st.expander("🔍 Contenuti utilizzati per la risposta"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Chunk {i+1}:**\n\n{doc.page_content}\n\n---")

                    # Risposta finale
                    st.markdown(BOT_TEMPLATE.format(msg=response["answer"]), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Errore durante la risposta: {e}")

    # Informativa rapida
    with st.expander("ℹ️ Note importanti"):
        st.markdown(
            """
- **Nessun limite di pagine/size**: l'elaborazione di PDF molto grandi può richiedere tempo e generare costi API più alti.
- **PDF scannerizzati** senza layer di testo potrebbero risultare vuoti: per tali file serve un OCR (opzionale, non incluso in questa versione).
- **Workspace isolata per sessione**: i tuoi indici non sono visibili ad altri utenti della stessa app.
- **Persistenza**: su hosting effimeri (es. Streamlit Cloud), i dati possono andare persi al riavvio; per persistenza reale usa uno storage esterno.
            """
        )


if __name__ == "__main__":
    main()
