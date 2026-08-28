import pandas as pd
import optuna
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Import your graph and node modules
from src.graph import app
import src.nodes as nodes


def objective(trial):
    # 1. Define the dynamic Pythonic search space
    top_k = trial.suggest_int("top_k", 10, 25)
    threshold = trial.suggest_float("threshold", -0.5, 0.5, step=0.25)
    final_docs = trial.suggest_int("final_docs", 7, 15)

    # 2. Patch the parameters in the retrieval node
    nodes.DENSE_LIMIT = top_k
    nodes.SPARSE_LIMIT = top_k
    nodes.RRF_LIMIT = top_k
    nodes.RERANK_THRESHOLD = threshold
    nodes.FINAL_DOCS = final_docs

    # Load a reproducible sample to keep trial execution fast
    df = pd.read_csv("./data/compliance_benchmark.csv").sample(n=5, random_state=42)

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)
    )

    metrics = [
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
        Faithfulness(llm=judge_llm),
    ]

    samples = []
    for _, row in df.iterrows():
        state = app.invoke(
            {
                "question": row["question"],
                "is_safe": True,
                "search_queries": [],
                "documents": [],
                "web_search_used": False,
                "generation": "",
                "audit_feedback": "",
                "is_compliant": False,
                "loop_count": 0,
            }
        )

        contexts = [doc.page_content for doc in state.get("documents", [])]
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                response=state.get("generation", ""),
                retrieved_contexts=contexts,
                reference=row["ground_truth"],
            )
        )

    eval_result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        run_config=RunConfig(max_workers=8, timeout=120, max_retries=3),
        raise_exceptions=False,
    )

    scores = eval_result.to_pandas().mean(numeric_only=True)

    context_precision = scores.get("context_precision", 0.0)
    context_recall = scores.get("context_recall", 0.0)
    faithfulness = scores.get("faithfulness", 0.0)

    # Log secondary metrics to the Optuna dashboard
    trial.set_user_attr("faithfulness", faithfulness)
    trial.set_user_attr("context_precision", context_precision)
    trial.set_user_attr("context_recall", context_recall)

    # Target metric: Maximize combined precision and recall
    return context_precision + context_recall


if __name__ == "__main__":
    print("\n--- STARTING OPTUNA HYPERPARAMETER SEARCH ---")

    # Create a study with persistent SQLite storage
    study = optuna.create_study(
        study_name="rag_optimization",
        direction="maximize",
        storage="sqlite:///rag_optuna.db",
        load_if_exists=True,
    )

    # Run the Bayesian optimization
    study.optimize(objective, n_trials=30)

    print("\n✅ Optuna Search Complete!")
    print(f"Best Trial: {study.best_trial.number}")
    print("Best Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
