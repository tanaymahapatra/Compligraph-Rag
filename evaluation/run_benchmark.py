import os
import re
import time
import random
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# ============================================================
# CONFIG
# ============================================================


# Gets the directory where the script lives (evaluation), then goes up one parent level, then into 'cache'
CACHE_DIR = str(Path(__file__).parent.parent / "cache")
OUTPUT_FILE = "./data/compliance_benchmark.csv"

NUM_QUESTIONS = 20
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
DELAY_SECONDS = 0


# ============================================================
# OUTPUT SCHEMA
# ============================================================


class QAPair(BaseModel):
    question: str = Field(
        description="A difficult reasoning question based only on the supplied regulatory text."
    )

    ground_truth: str = Field(
        description="The factually correct answer supported only by the supplied regulatory text."
    )


# ============================================================
# CLEAN TEXT
# ============================================================


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"<--\s*\*?\s*Image\s*\*?\s*-->",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"!\[[^\]]*\]\([^)]+\)",
        " ",
        text,
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# LOAD + SPLIT DOCUMENTS
# ============================================================


def load_chunks():

    if not os.path.isdir(CACHE_DIR):
        raise FileNotFoundError(f"Cache directory not found: {CACHE_DIR}")

    loader = DirectoryLoader(
        CACHE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"No Markdown files found in {CACHE_DIR}")

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    documents = [doc for doc in documents if len(doc.page_content) > 200]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 200]

    print(f"Loaded {len(documents)} documents → " f"{len(chunks)} usable passages")

    return chunks


# ============================================================
# GENERATE BENCHMARK
# ============================================================


def generate_custom_benchmark(
    num_questions=NUM_QUESTIONS,
    delay_seconds=DELAY_SECONDS,
):

    chunks = load_chunks()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        temperature=0.0,
        max_retries=5,
    )

    qa_llm = llm.with_structured_output(QAPair)

    random.seed(42)
    random.shuffle(chunks)

    questions = []
    used_questions = set()

    print(f"\nGenerating {num_questions} benchmark questions...")

    for i, chunk in enumerate(chunks):

        if len(questions) >= num_questions:
            break

        print(
            f"\n[{len(questions) + 1}/{num_questions}] "
            f"Processing passage {i + 1}/{len(chunks)}"
        )

        prompt = f"""
You are creating a high-quality benchmark for a
financial regulatory RAG system.

Generate ONE difficult question and its ground-truth answer
using ONLY the regulatory passage below.

Rules:

1. The answer must be completely supported by the passage.
2. Do not use outside knowledge.
3. Do not invent facts, numbers, dates, entities, or requirements.
4. The question should require reasoning rather than simple copying.
5. Prefer requirements, thresholds, conditions, exceptions,
   comparisons, dates, limits, or regulatory consequences.
6. Mention the relevant regulation, authority, institution,
   document, or subject when the passage makes this possible.
7. Avoid vague wording such as "According to the text..."
8. The ground truth must directly answer the question.
9. Do not create a question if the passage does not contain
   enough information to answer it reliably.

REGULATORY PASSAGE:

{chunk.page_content}
"""

        try:

            result = qa_llm.invoke(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )

            question = result.question.strip()
            answer = result.ground_truth.strip()

            if not question or not answer:
                print("⚠️ Empty result. Skipping.")
                continue

            # Prevent duplicate questions.
            question_key = re.sub(
                r"\s+",
                " ",
                question.lower(),
            )

            if question_key in used_questions:
                print("⚠️ Duplicate question. Skipping.")
                continue

            used_questions.add(question_key)

            questions.append(
                {
                    "question": question,
                    "ground_truth": answer,
                    "context": chunk.page_content,
                }
            )

            print("✅ Generated")

            if delay_seconds > 0 and len(questions) < num_questions:
                time.sleep(delay_seconds)

        except Exception as e:

            print(f"❌ Generation failed: {e}")

            if delay_seconds:
                time.sleep(delay_seconds)

    # ========================================================
    # SAVE
    # ========================================================

    if not questions:
        print("\n❌ No benchmark questions generated.")
        return

    os.makedirs("./data", exist_ok=True)

    df = pd.DataFrame(questions)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\n✅ Generated {len(df)} benchmark questions.")

    print(f"Saved to: {OUTPUT_FILE}")

    print("\nPreview:")
    print(df[["question", "ground_truth"]].head().to_string(index=False))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_custom_benchmark(
        num_questions=20,
        delay_seconds=0,
    )
