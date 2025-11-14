#fortnitecompiler.py
import os, io, uuid, tempfile, shutil
import streamlit as st
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Fortnite Compiler (FT. Ollama)", layout="wide")
#fortnite
def embedder(model: str):
    return OllamaEmbeddings(model=model)

def llm(model: str, temperature: float = 0.2):
    return ChatOllama(model=model, temperature=temperature)

def read_pdfs(files: List[io.BytesIO], doc_ids: List[str]) -> List[Document]:
    docs: List[Document] = []
    for f, did in zip(files, doc_ids):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            try:
                f.seek(0)
            except Exception:
                pass
            shutil.copyfileobj(f, tmp)
            tmp.flush()
            loader = PyPDFLoader(tmp.name)
            pages = loader.load()
        for p in pages:
            p.metadata = {**(p.metadata or {}), "doc_id": did, "parent_id": f"{did}/page-{p.metadata.get('page', 0)}"}
        docs.extend(pages)
    return docs

def section_split(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ","\n### ","\n\n","\n"," "],
        add_start_index=True,
    )
    out: List[Document] = []
    for page in docs:
        parts = splitter.split_text(page.page_content)
        for i, t in enumerate(parts):
            out.append(Document(
                page_content=t,
                metadata={
                    "doc_id": page.metadata.get("doc_id"),
                    "page": page.metadata.get("page", 0),
                    "parent_id": page.metadata.get("parent_id"),
                    "section_id": f'{page.metadata.get("parent_id")}/sec-{i:04d}',
                    "source": page.metadata.get("source", "pdf")
                }
            ))
    return out

def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool):
    existing = [c.name for c in client.get_collections().collections]
    if recreate and name in existing:
        client.delete_collection(name)
        existing.remove(name)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

def batched(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def index_docs(vs: LCQdrant, docs: List[Document], batch_size: int):
    for batch in batched(docs, batch_size):
        vs.add_documents(batch)

def lc_qdrant(client: QdrantClient, name: str, embeddings: OllamaEmbeddings) -> LCQdrant:
    return LCQdrant(client=client, collection_name=name, embeddings=embeddings)

def format_ctx(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[{i}] {d.metadata.get('doc_id')} p{d.metadata.get('page')} :: {d.page_content}")
    return "\n\n".join(blocks)

def build_chain(vs: LCQdrant, llm_model: str):
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 40})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context. Cite as [n]. If unknown, say so."),
        ("human", "Question:\n{q}\n\nContext:\n{ctx}\n\nAnswer concisely with citations.")
    ])
    model = llm(llm_model)
    chain = (
        {"ctx": retriever | RunnableLambda(format_ctx), "q": RunnablePassthrough()}
        | prompt | model | StrOutputParser()
    )
    return chain, retriever

def hierarchical_expand(client: QdrantClient, collection: str, seeds: List[Document], max_sections: int, sibling_limit: int) -> List[Document]:
    ids = {d.metadata.get("section_id") for d in seeds if d.metadata.get("section_id")}
    out: Dict[str, Document] = {d.metadata["section_id"]: d for d in seeds if d.metadata.get("section_id")}
    parents = list({d.metadata.get("parent_id") for d in seeds if d.metadata.get("parent_id")})
    for parent in parents:
        scroll, _ = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="parent_id", match=models.MatchValue(value=parent))]),
            limit=sibling_limit,
            with_payload=True,
            with_vectors=False,
        )
        for pt in scroll:
            sid = pt.payload.get("section_id")
            if not sid or sid in ids:
                continue
            out[sid] = Document(
                page_content=pt.payload.get("page_content",""),
                metadata={k: pt.payload.get(k) for k in ["doc_id","page","parent_id","section_id","source"] if k in pt.payload}
            )
            if len(out) >= max_sections:
                break
        if len(out) >= max_sections:
            break
    return list(out.values())

def make_context(client: QdrantClient, collection: str, retriever, query: str, expand_sections: int, sibling_limit: int) -> List[Document]:
    base_docs: List[Document] = retriever.invoke(query)
    return hierarchical_expand(client, collection, base_docs, expand_sections, sibling_limit)

st.sidebar.header("Settings")
qdrant_url = st.sidebar.text_input("Qdrant URL", "http://127.0.0.1:6333")
collection = st.sidebar.text_input("Collection", "pdf_rag_sections")
embed_model = st.sidebar.text_input("Ollama embedding model", "nomic-embed-text")
llm_model = st.sidebar.text_input("Ollama LLM", "mistral:latest")
chunk_size = st.sidebar.number_input("Chunk size", 256, 4000, 1200, 64)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 1000, 200, 16)
batch_size = st.sidebar.number_input("Index batch size", 8, 2048, 128, 8)
recreate = st.sidebar.checkbox("Recreate collection", False)
expand_sections = st.sidebar.number_input("Max expanded sections per query", 6, 200, 24, 2)
sibling_limit = st.sidebar.number_input("Sibling limit per parent", 2, 200, 8, 1)
temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.2, 0.1)

st.title("Fortnite Compiler (FT. Ollama)")

if "client" not in st.session_state:
    st.session_state.client = QdrantClient(url=qdrant_url)
if "emb" not in st.session_state or st.session_state.get("emb_name") != embed_model:
    st.session_state.emb = embedder(embed_model)
    st.session_state.emb_name = embed_model
if "vs" not in st.session_state or st.session_state.get("collection") != collection or recreate:
    test_vec_dim = len(st.session_state.emb.embed_query("test"))
    ensure_collection(st.session_state.client, collection, dim=test_vec_dim, recreate=recreate)
    st.session_state.vs = lc_qdrant(st.session_state.client, collection, st.session_state.emb)
    st.session_state.collection = collection
if "chain" not in st.session_state or st.session_state.get("llm_name") != llm_model:
    st.session_state.chain, st.session_state.retriever = build_chain(st.session_state.vs, llm_model)
    st.session_state.llm_name = llm_model

st.subheader("Ingest PDFs")
files = st.file_uploader("Drop PDFs", type=["pdf"], accept_multiple_files=True)
if st.button("Ingest") and files:
    ids = [f.name for f in files]
    pages = read_pdfs(files, ids)
    chunks = section_split(pages, chunk_size, chunk_overlap)
    with st.status(f"Indexing {len(chunks)} chunks…", expanded=True) as status:
        for b in batched(chunks, int(batch_size)):
            st.session_state.vs.add_documents(b)
        status.update(label="Done", state="complete")
    st.success("Indexed")
#intake questions for fortnite
st.subheader("Ask")
q = st.text_input("Question")
if st.button("Search & Answer") and q:
    base = st.session_state.retriever.invoke(q)
    expanded = hierarchical_expand(st.session_state.client, collection, base, int(expand_sections), int(sibling_limit))
    ctx_text = format_ctx(expanded)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context. Cite as [n]. If unknown, say so."),
        ("human", "Question:\n{q}\n\nContext:\n{ctx}\n\nAnswer concisely with citations.")
    ])
    ans = (prompt | llm(llm_model, temperature) | StrOutputParser()).invoke({"q": q, "ctx": ctx_text})
    st.markdown("### Answer")
    st.write(ans)
    with st.expander("Context"):
        st.text(ctx_text)

