import os
import time
import pandas as pd
from dotenv import load_dotenv

from src.graph import app

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmCategory,
    HarmBlockThreshold,
)

load_dotenv()

CSV_PATH = "./data/compliance_benchmark.csv"
REPORT_PATH = "./data/final_evaluation_report.csv"
DELAY = 0


def get_value(state, key, default=None):
    if hasattr(state, key):
        return getattr(state, key, default)
    return state.get(key, default)


def get_contexts(documents):
    contexts = []

    for doc in documents or []:
        if hasattr(doc, "page_content"):
            text = doc.page_content
        elif isinstance(doc, dict):
            text = (
                doc.get("page_content") or doc.get("text") or doc.get("content") or ""
            )
        else:
            text = str(doc)

        if text and text.strip():
            contexts.append(text.strip())

    return contexts


def run_graph(question):
    state = {
        "question": question,
        "is_safe": False,
        "search_queries": [],
        "documents": [],
        "generation": "",
        "web_search_used": False,
        "audit_feedback": "",
        "is_compliant": False,
        "loop_count": 0,
    }

    result = app.invoke(state)

    answer = get_value(result, "generation", "")
    documents = get_value(result, "documents", [])

    return str(answer).strip(), get_contexts(documents), result


def create_metrics():
    # Set safety thresholds to BLOCK_NONE to prevent Gemini API from blocking regulatory text evaluation prompts
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.OFF,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.OFF,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.OFF,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.OFF,
    }
    
    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.7-flash",
            temperature=0,
            max_retries=5,
            safety_settings=safety_settings,
        )
    )

    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            max_retries=5,
        )
    )

    return [
        Faithfulness(llm=judge_llm),
        AnswerCorrectness(
            llm=judge_llm,
            embeddings=judge_embeddings,
        ),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]


def run_evaluation(
    csv_path=CSV_PATH,
    report_path=REPORT_PATH,
    delay=DELAY,
):
    print("\n--- LOADING DATASET ---")

    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    required = {"question", "ground_truth"}

    if not required.issubset(df.columns):
        print("❌ CSV must contain: 'question' and 'ground_truth'")
        return

    samples = []
    records = []

    print(f"Loaded {len(df)} benchmark questions.")

    print("\n--- RUNNING GRAPH ---")

    for i, row in df.iterrows():

        question = str(row["question"]).strip()
        reference = str(row["ground_truth"]).strip()

        print(f"\n[{i + 1}/{len(df)}] {question}")

        try:
            answer, contexts, state = run_graph(question)

            is_safe = get_value(state, "is_safe", True)
            is_compliant = get_value(state, "is_compliant", False)

            if not is_safe:
                print("⚠️ REJECTED BY GUARDRAIL! Retriever skipped.")

            samples.append(
                SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                    reference=reference,
                )
            )

            records.append(
                {
                    "question": question,
                    "generation": answer,
                    "ground_truth": reference,
                    "num_contexts": len(contexts),
                    "is_safe": is_safe,
                    "is_compliant": is_compliant,
                    "web_search_used": get_value(
                        state,
                        "web_search_used",
                        False,
                    ),
                    "loop_count": get_value(
                        state,
                        "loop_count",
                        0,
                    ),
                }
            )

            print(f"Contexts: {len(contexts)} | " f"Audited: {is_compliant}")

        except Exception as e:
            print(f"❌ Failed: {e}")

        if i < len(df) - 1 and delay:
            print(f"Sleeping {delay}s...")
            time.sleep(delay)

    if not samples:
        print("❌ No successful samples.")
        return

    print(f"\n--- EVALUATING {len(samples)} SAMPLES ---")

    dataset = EvaluationDataset(samples=samples)

    metrics = create_metrics()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(
            max_workers=1,
            timeout=120,
            max_retries=5,
        ),
        raise_exceptions=False,
    )

    print("\n--- EXPORTING REPORT ---")

    results_df = result.to_pandas()
    metadata_df = pd.DataFrame(records)

    if len(results_df) == len(metadata_df):
        for column in [
            "num_contexts",
            "is_safe",
            "is_compliant",
            "web_search_used",
            "loop_count",
        ]:
            results_df[column] = metadata_df[column].values

    os.makedirs(
        os.path.dirname(report_path) or ".",
        exist_ok=True,
    )

    results_df.to_csv(
        report_path,
        index=False,
    )

    print(f"\n✅ Report saved to: {report_path}")

    print("\n--- SCORES ---")

    for metric in [
        "faithfulness",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]:
        if metric in results_df:
            score = pd.to_numeric(
                results_df[metric],
                errors="coerce",
            ).mean()

            print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    run_evaluation()
