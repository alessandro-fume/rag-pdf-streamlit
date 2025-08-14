import os
import re
import json
import pathlib
import shutil
from datetime import datetime

import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# NEW: auth
import streamlit_authenticator as stauth

# =============================
# Branding / impostazioni base
# =============================
APP_TITLE = "Chatbot EOS Reply – PDF Documents"
FAVICON_PATH = "logo_favicon.png"   # favicon (quadrata)
LOGO_PATH   = "logo_eos_reply.png"  # logo in pagina/sidebar

# === Multi-tenant root (ogni utente ha il suo "spazio") ===
TENANT_ROOT = "tenants"             # dentro qui creiamo sottocartella per utente
TRUST_MARK_FILE = ".trusted_by_app"
TRUST_MARK_VALUE = "EOS-REPLY-RAG-V1"
LEGACY_PREFIXES = ("ws_",)

# Sottocartelle per ogni utente
VSTORE_DIRNAME = "vectorstore"
CONV_SUBDIR    = "conversations"    # per-insieme di chat: tenants/<user>/vectorstore/<index>/conversations/*.json

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
# Utility comuni
# =============================
def get_openai_key_from_secrets() -> str:
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("❌ Nessuna chiave API trovata. Imposta st.secrets['OPENAI_API_KEY'].")
        st.stop()
    os.environ["OPENAI_API_KEY"] = key
    return key

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s[:80] or "user"

def user_root(user_slug: str) -> str:
    root = os.path.join(TENANT_ROOT, user_slug)
    os.makedirs(root, exist_ok=True)
    return root

def ensure_store_dir(user_slug: str) -> str:
    """Ritorna la cartella vectorstore per l'utente loggato."""
    base = os.path.join(user_root(user_slug), VSTORE_DIRNAME)
    os.makedirs(base, exist_ok=True)
    return base

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

def list_indices(user_slug: str) -> list:
    base = ensure_store_dir(user_slug)
    items = []
    if os.path.exists(base):
        for d in os.scandir(base):
            if d.is_dir() and (_has_trust_mark(d.path) or any(d.name.startswith(p) for p in LEGACY_PREFIXES)):
                items.append(d.name)
    return sorted(items)

# ======= Persistenza conversazioni (per indice & utente) =======
def ensure_conv_dir(user_slug: str, index: str) -> str:
    conv_dir = os.path.join(ensure_store_dir(user_slug), index, CONV_SUBDIR)
    os.makedirs(conv_dir, exist_ok=True)
    return conv_dir

def conv_file_path(user_slug: str, index: str, conv_id: str) -> str:
    return os.path.join(ensure_conv_dir(user_slug, index), f"{conv_id}.json")

def new_conv_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def list_conversations(user_slug: str, index: str) -> list[tuple[str, str]]:
    conv_dir = ensure_conv_dir(user_slug, index)
    items = []
    for f in os.scandir(conv_dir):
        if f.is_file() and f.name.endswith(".json"):
            conv_id = f.name[:-5]
            label = conv_id.replace("_", " ").replace("-", ":")
            items.append((conv_id, label))
    return sorted(items, key=lambda x: x[0], reverse=True)

def load_chat(user_slug: str, index: str, conv_id: str) -> list[dict]:
    p = conv_file_path(user_slug, index, conv_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat(user_slug: str, index: str, conv_id: str, messages: list[dict]) -> None:
    p = conv_file_path(user_slug, index, conv_id)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def safe_rerun():
    """Compat per Streamlit versioni diverse."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# =============================
# PDF → Documents con metadati pagina
# =============================
def documents_from_pdf_with_pages(uploaded_pdf) -> list[Document]:
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vs = FAISS.from_documents(docs, embedding=embeddings)
    os.makedirs(path, exist_ok=True)
    vs.save_local(path)
    try:
        with open(os.path.join(path, TRUST_MARK_FILE), "w", encoding="utf-8") as f:
            f.write(TRUST_MARK_VALUE)
    except Exception:
        pass
    return vs

@st.cache_resource(show_spinner=False)
def load_vectorstore(path: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def build_chain(vectorstore, temperature: float):
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-3.5-turbo",
        max_tokens=900
    )
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
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
# AUTH helpers
# =============================
def _build_auth_params_from_secrets():
    """Converte st.secrets (read-only) in dict/list mutabili per streamlit-authenticator."""
    auth_sec = st.secrets.get("auth", None)
    if not auth_sec:
        st.error("Configurazione autenticazione assente in st.secrets['auth'].")
        st.stop()

    # credentials.usernames -> dict mutabile
    usernames = {}
    creds = auth_sec.get("credentials", {}).get("usernames", {})
    for uname, u in creds.items():
        usernames[str(uname)] = {
            "name": str(u["name"]),
            "email": str(u["email"]),
            "password": str(u["password"]),
        }
    credentials = {"usernames": usernames}

    cookie_name = str(auth_sec.get("cookie_name", "eos_reply_chatbot"))
    cookie_key  = str(auth_sec.get("cookie_key",  "PLEASE_SET_A_RANDOM_SECRET"))
    cookie_expiry_days = int(auth_sec.get("cookie_expiry_days", 30))

    preauthorized = {}
    if "preauthorized" in auth_sec:
        emails = [str(e) for e in auth_sec["preauthorized"].get("emails", [])]
        preauthorized = {"emails": emails}

    return credentials, cookie_name, cookie_key, cookie_expiry_days, preauthorized

# =============================
# AUTH
# =============================
def do_login():
    st.set_page_config(page_title=APP_TITLE, page_icon=FAVICON_PATH)
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()

    # Costruisci parametri mutabili dai secrets
    credentials, cookie_name, cookie_key, cookie_expiry_days, preauthorized = _build_auth_params_from_secrets()

    # Istanzia Authenticate con compatibilità firme diverse
    try:
        authenticator = stauth.Authenticate(
            credentials=credentials,
            cookie_name=cookie_name,
            key=cookie_key,
            cookie_expiry_days=cookie_expiry_days,
            preauthorized=preauthorized
        )
    except TypeError:
        cookie = {"name": cookie_name, "key": cookie_key, "expiry_days": cookie_expiry_days}
        try:
            authenticator = stauth.Authenticate(credentials, cookie, preauthorized)
        except TypeError:
            authenticator = stauth.Authenticate(credentials, cookie_name, cookie_key, cookie_expiry_days, preauthorized)

    # --- Login (compat con versioni nuove/vecchie) ---
    result = None
    try:
        result = authenticator.login(
            location="main",
            fields={"Form name": "Login", "Username": "Username", "Password": "Password"},
            key="Login"
        )
    except TypeError:
        # vecchie firme che non accettano 'fields' o keyword
        try:
            result = authenticator.login("main")
        except TypeError:
            # antichissima: ('Login','main')
            result = authenticator.login("Login", "main")

    if isinstance(result, tuple):
        # Versioni che ritornano (name, auth_status, username)
        name, auth_status, username = result
    else:
        # Versioni che non ritornano nulla e scrivono in session_state
        name = st.session_state.get("name")
        auth_status = st.session_state.get("authentication_status")
        username = st.session_state.get("username")

    if auth_status is False:
        st.error("Username/Password non corretti.")
        st.stop()
    if auth_status is None:
        st.info("Inserisci le credenziali per accedere.")
        st.stop()

    # Logged in
    authenticator.logout("Logout", "sidebar")
    return name, username, authenticator

# =============================
# App principale (post-login)
# =============================
def main_app(user_name: str, user_username: str):
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()
    get_openai_key_from_secrets()

    # Slug unico per utente (cartelle separate)
    user_slug = slugify(user_username or user_name or "user")

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
        "conv_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ---------- Sidebar ----------
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=120)
        st.caption(f"👤 Utente: **{user_name}**")

        st.subheader("⚙️ Impostazioni")
        st.session_state.llm_temperature = st.slider("Temperatura del modello", 0.0, 1.0, 0.5, step=0.1)

        st.subheader("📁 I miei documenti indicizzati")
        indices = list_indices(user_slug)
        selected = st.selectbox("Scegli un documento indicizzato", options=["-- Nessuno --"] + indices)

        # Archivio conversazioni per indice
        if selected != "-- Nessuno --":
            with st.expander("🗂️ Archivio conversazioni"):
                convs = list_conversations(user_slug, selected)
                if not convs:
                    st.caption("Nessuna conversazione salvata per questo documento.")
                else:
                    labels = [lbl for _, lbl in convs]
                    ids    = [cid for cid, _ in convs]
                    try:
                        default_ix = ids.index(st.session_state.conv_id) if st.session_state.conv_id in ids else 0
                    except Exception:
                        default_ix = 0
                    pick_ix = st.selectbox("Seleziona una conversazione:", options=range(len(ids)),
                                           format_func=lambda i: labels[i], index=default_ix)
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("Apri"):
                            st.session_state.conv_id = ids[pick_ix]
                            st.session_state.messages = load_chat(user_slug, selected, st.session_state.conv_id)
                            safe_rerun()
                    with col_b:
                        if st.button("Nuova"):
                            st.session_state.conv_id = new_conv_id()
                            st.session_state.messages = []
                            save_chat(user_slug, selected, st.session_state.conv_id, st.session_state.messages)
                            safe_rerun()
                    with col_c:
                        if st.button("Elimina"):
                            p = conv_file_path(user_slug, selected, ids[pick_ix])
                            try:
                                os.remove(p)
                                if st.session_state.conv_id == ids[pick_ix]:
                                    st.session_state.conv_id = None
                                    st.session_state.messages = []
                                safe_rerun()
                            except Exception as e:
                                st.error(f"Errore eliminazione: {e}")

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
                        shutil.rmtree(os.path.join(ensure_store_dir(user_slug), selected))
                        st.success(f"Indice '{selected}' eliminato.")
                        st.session_state.update({
                            "conversation": None,
                            "last_text_preview": "",
                            "current_index": None,
                            "confirm_delete": False,
                            "messages": [],
                            "last_sources": [],
                            "conv_id": None,
                        })
                        safe_rerun()
                    except Exception as e:
                        st.error(f"Errore durante l'eliminazione: {e}")
                        st.session_state.confirm_delete = False

        # Upload e indicizzazione
        st.markdown("---")
        st.subheader("📤 Carica nuovo PDF da indicizzare (privato)")
        pdf_doc = st.file_uploader("Carica un PDF", type=["pdf"], accept_multiple_files=False)
        overwrite = st.checkbox("Sovrascrivi se già esistente", value=False)

        if pdf_doc and st.button("Processa e indicizza"):
            with st.spinner("📚 Indicizzazione in corso..."):
                try:
                    filename = safe_name(pdf_doc.name)
                    vectorstore_path = os.path.join(ensure_store_dir(user_slug), filename)
                    if os.path.exists(vectorstore_path) and not overwrite:
                        st.warning("Documento già processato.")
                        st.stop()

                    docs = documents_from_pdf_with_pages(pdf_doc)
                    if not docs:
                        st.error("❌ Nessun testo leggibile trovato nel PDF (forse scannerizzato senza OCR).")
                        st.stop()

                    vs = build_vectorstore_from_documents(docs, vectorstore_path)

                    try:
                        with open(os.path.join(vectorstore_path, "documento.pdf"), "wb") as f:
                            f.write(pdf_doc.getbuffer())
                    except Exception:
                        pass

                    preview_text = "\n".join([d.page_content for d in docs[:4]])
                    st.session_state.last_text_preview = preview_text[:3000]
                    st.session_state.conversation = build_chain(vs, st.session_state.llm_temperature)
                    st.session_state.current_index = filename
                    st.session_state.messages = []
                    st.session_state.last_sources = []
                    st.session_state.conv_id = new_conv_id()
                    save_chat(user_slug, st.session_state.current_index, st.session_state.conv_id, st.session_state.messages)

                    st.success(f"✅ Documento '{filename}' indicizzato e pronto all'uso!")
                except Exception as e:
                    st.error(f"❌ Errore durante l'elaborazione: {e}")

    # ---------- Caricamento indice esistente ----------
    if selected != "-- Nessuno --" and selected != st.session_state.current_index:
        with st.spinner("🔁 Caricamento indice..."):
            try:
                vs = load_vectorstore(os.path.join(ensure_store_dir(user_slug), selected))
                st.session_state.conversation = build_chain(vs, st.session_state.llm_temperature)
                st.session_state.current_index = selected

                pdf_path = os.path.join(ensure_store_dir(user_slug), selected, "documento.pdf")
                if os.path.exists(pdf_path):
                    try:
                        with pdfplumber.open(pdf_path) as pdf_file:
                            text = "".join([p.extract_text() or "" for p in pdf_file.pages])
                            st.session_state.last_text_preview = text[:3000]
                    except Exception:
                        st.session_state.last_text_preview = ""

                convs = list_conversations(user_slug, selected)
                if convs:
                    st.session_state.conv_id = convs[0][0]
                    st.session_state.messages = load_chat(user_slug, selected, st.session_state.conv_id)
                else:
                    st.session_state.conv_id = new_conv_id()
                    st.session_state.messages = []
                    save_chat(user_slug, selected, st.session_state.conv_id, st.session_state.messages)

                st.session_state.last_sources = []
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

    # ---------- Input domanda ----------
    with st.form("qa_form", clear_on_submit=True):
        user_question = st.text_input("Fai una domanda sul documento selezionato:")
        submitted = st.form_submit_button("Invia")

    if submitted and user_question and user_question.strip():
        if not st.session_state.conversation:
            st.warning("⚠️ Seleziona o carica prima un documento.")
        else:
            with st.spinner("🧠 Elaborazione..."):
                try:
                    q = user_question.strip()
                    st.session_state.messages.append({"role": "user", "content": q})
                    response = st.session_state.conversation({"question": q})
                    answer = response["answer"]
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_sources = response.get("source_documents", [])
                    # persist
                    if st.session_state.current_index and st.session_state.conv_id:
                        save_chat(user_slug, st.session_state.current_index, st.session_state.conv_id, st.session_state.messages)
                except Exception as e:
                    st.error(f"❌ Errore durante la risposta: {e}")

    # ---------- Render chat ----------
    if st.session_state.get("messages"):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(USER_TEMPLATE.format(msg=msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(BOT_TEMPLATE.format(msg=msg["content"]), unsafe_allow_html=True)

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

    with st.expander("ℹ️ Note importanti"):
        st.markdown(
            """
- Ogni utente vede SOLO i propri documenti/conversazioni (spazi isolati).
- I PDF scannerizzati (solo immagine) potrebbero non contenere testo ricercabile.
- Le conversazioni sono salvate localmente nel tuo spazio utente.
            """
        )

# =============================
# Entrypoint
# =============================
def main():
    # Login
    name, username, _auth = do_login()
    # App per l'utente autenticato
    main_app(name, username)

if __name__ == "__main__":
    main()
