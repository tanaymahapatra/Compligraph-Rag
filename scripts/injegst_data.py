import os
import uuid
import re
from pathlib import Path
from bs4 import BeautifulSoup

from docling.document_converter import DocumentConverter
from fastembed import SparseTextEmbedding, TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "compligraph_docs"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# 🔴 Set to True to guarantee we wipe the corrupted cache and apply new regex!
FORCE_REPARSE = True

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING & PRE-PROCESSING
# ============================================================


def clean_html_noise(file_path: Path) -> Path:
    """Pre-cleans HTML files using BeautifulSoup to remove SEC boilerplate before Docling sees it."""
    if file_path.suffix.lower() not in {".html", ".htm"}:
        return file_path

    print(f"🧹 Pre-cleaning HTML boilerplate: {file_path.name}")
    raw_html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    # Strip non-content noise (scripts, styles, headers, footers)
    for element in soup(["script", "style", "nav", "header", "footer", "meta"]):
        element.decompose()

    clean_html = str(soup)

    # Save to a temporary clean file to feed to Docling
    temp_file = CACHE_DIR / f"clean_{file_path.name}"
    temp_file.write_text(clean_html, encoding="utf-8")
    return temp_file


def clean_text(text: str) -> str:
    """Post-cleans the Docling markdown output."""
    if not isinstance(text, str):
        return ""

    # Remove PDF image tags and URLs
    text = re.sub(r"\|?<-- Image -->\|?", " ", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"https?://[^\s]+|www\.[^\s]+", " ", text)

    # --- NEW ARTIFACT FILTERS ---
    # Strip Docling's failed formula tags
    text = re.sub(r"<\|--.*?--\|>", " ", text)

    # Strip PDF font encoding gibberish (e.g., G84G97G98...)
    text = re.sub(r"(?:G\d{2,3}){3,}", " ", text)
    # ----------------------------

    # Normalize excessive spaces/newlines without breaking table pipes (|)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# PARSE DOCUMENTS
# ============================================================


def parse_documents(force_reparse: bool = False):
    converter = DocumentConverter()
    documents = {}

    files = [
        file
        for file in DATA_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in {".pdf", ".html", ".htm"}
    ]

    if not files:
        raise FileNotFoundError(f"No PDF or HTML files found in {DATA_DIR}")

    print(f"Found {len(files)} document(s) in {DATA_DIR}: {[f.name for f in files]}")

    for file in files:
        cache_file = CACHE_DIR / f"{file.stem}.md"
        cache_valid = (
            cache_file.exists()
            and not force_reparse
            and len(cache_file.read_text(encoding="utf-8").strip()) > 50
        )

        if cache_valid:
            print(f"⚡ Loading cache: {file.name} ({cache_file.name})")
            text = cache_file.read_text(encoding="utf-8")
            text = clean_text(text)
        else:
            print(
                f"⚙️ Parsing with Docling: {file.name} (this may take a few seconds)..."
            )

            try:
                # Pre-clean HTML if necessary
                target_file = clean_html_noise(file)

                result = converter.convert(str(target_file))
                raw_text = result.document.export_to_markdown()

                # Apply regex cleaning BEFORE caching
                text = clean_text(raw_text)

                if not text or len(text.strip()) == 0:
                    print(f"⚠️ Warning: Docling extracted empty text from {file.name}")
                    continue

                cache_file.write_text(text, encoding="utf-8")
                print(f"✅ Cached clean markdown: {cache_file.name}")

            except Exception as e:
                print(f"❌ Failed to parse {file.name}: {e}")
                continue

        documents[file.name] = text

    return documents


# ============================================================
# CHUNK DOCUMENTS
# ============================================================


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )

    chunks = []

    for filename, text in documents.items():
        raw_chunks = splitter.split_text(text)

        # Add document type to metadata for retrieval filtering
        doc_type = "html" if filename.lower().endswith(("htm", "html")) else "pdf"

        for index, chunk in enumerate(raw_chunks):
            if not chunk.strip():
                continue

            # Drop chunks that are purely noise
            alphanumeric_count = len(re.sub(r"[^a-zA-Z0-9]", "", chunk))
            if alphanumeric_count < 30:
                continue

            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "source_file": filename,
                    "doc_type": doc_type,
                    "chunk_index": index,
                }
            )

    print(f"Created {len(chunks)} total chunks across all documents.")
    return chunks


# ============================================================
# INGEST
# ============================================================


def ingest(chunks):
    if not chunks:
        print("❌ No valid chunks to ingest. Aborting.")
        return

    print("\nInitializing embedding models...")

    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    texts = [chunk["text"] for chunk in chunks]

    dummy_dense = next(dense_model.embed(["dummy text"]))
    dense_dimension = len(dummy_dense)

    print(f"Creating hybrid collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dimension, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )

    print("Streaming embeddings and uploading to Qdrant...")
    dense_generator = dense_model.embed(texts)
    sparse_generator = sparse_model.embed(texts)

    batch_size = 100
    points_batch = []
    total_uploaded = 0

    for chunk, dense, sparse in zip(chunks, dense_generator, sparse_generator):
        points_batch.append(
            models.PointStruct(
                id=chunk["id"],
                vector={
                    "dense": dense.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload={
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "doc_type": chunk["doc_type"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
        )

        if len(points_batch) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            total_uploaded += len(points_batch)
            print(f"Uploaded {total_uploaded}/{len(chunks)}")
            points_batch = []

    if points_batch:
        client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
        total_uploaded += len(points_batch)
        print(f"Uploaded {total_uploaded}/{len(chunks)}")

    info = client.get_collection(COLLECTION_NAME)
    print("\n" + "=" * 60)
    print("QDRANT COLLECTION VERIFIED")
    print("=" * 60)
    print(f"Total Points Indexed: {info.points_count}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COMPLIGRAPH HYBRID INGESTION (REBUILT)")
    print("=" * 60)

    documents = parse_documents(force_reparse=FORCE_REPARSE)
    chunks = create_chunks(documents)
    ingest(chunks)

    print("\n✅ Hybrid ingestion completed successfully.")
