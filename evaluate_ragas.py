"""
Опциональная ручная оценка качества RAG через RAGAS.

Датасет (вопросы и эталонные ответы) принадлежит вертикали и задаётся явно: --dataset PATH
(например, examples/equipment_manual/eval.json). Встроенного набора вопросов нет.

Судья RAGAS по умолчанию — тот же endpoint, который настроен для самого приложения (LLM_API_KEY /
LLM_BASE_URL): локальная конфигурация даёт локального судью и ничего не уходит в hosted OpenAI из-за
случайно заданного OPENAI_API_KEY. Другой судья включается только явно и полностью: RAG_EVAL_JUDGE_BASE_URL +
RAG_EVAL_JUDGE_API_KEY (подробнее в app_core/evaluation/config.py). Каждый запуск делает платные/долгие вызовы
модели-судьи; это не часть CI и не quality gate.

Context Precision оценивает ретрив относительно эталона из датасета, а не обрезки ответа модели.
Context Utilization — насколько извлечённый контекст полезен для фактического ответа RAG.
"""

import math
import os
import sys

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from app_core.config.env import load_repo_env
from app_core.evaluation.config import (
    EvalConfigError,
    JudgeConfig,
    apply_privacy_defaults,
    parse_args,
    resolve_judge_config,
)
from app_core.evaluation.dataset import EvalDataset, EvalDatasetError, load_eval_dataset
from rag_pipeline import RAGPipeline

# Метрики и run_config (RAGAS 0.2+)
try:
    from ragas.metrics._context_precision import ContextPrecision, ContextUtilization
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    _METRICS_MODE = "v2"
except ImportError:
    _METRICS_MODE = "legacy"
    try:
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        from ragas.metrics import faithfulness, context_precision


def _build_metrics():
    if _METRICS_MODE == "v2":
        return [
            Faithfulness(max_retries=3),
            ContextPrecision(max_retries=3),
            ContextUtilization(max_retries=3),
        ], RunConfig(timeout=180, max_retries=10, max_workers=8)
    return [faithfulness(), context_precision()], None


def _metric_keys(metrics):
    if _METRICS_MODE == "v2":
        return ["faithfulness", "context_precision", "context_utilization"]
    return ["faithfulness", "context_precision"]


def build_judge_models(judge: JudgeConfig):
    """
    LangChain chat/embedding clients bound to the judge endpoint, wrapped for RAGAS. No request is made here.

    The explicit `llm=` / `embeddings=` also stop RAGAS from creating its own default OpenAI clients.
    """
    chat = ChatOpenAI(
        model=judge.model,
        base_url=judge.base_url,
        api_key=judge.api_key,
        temperature=0,
        timeout=float(os.getenv("OPENAI_TIMEOUT", "180")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "5")),
    )
    embeddings = OpenAIEmbeddings(
        model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=judge.base_url,
        api_key=judge.api_key,
        check_embedding_ctx_length=False,  # OpenAI-compatible local endpoints do not accept token arrays
    )
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def prepare_dataset(pipeline: RAGPipeline, spec: EvalDataset) -> Dataset:
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("[*] Получение ответов от RAG системы...\n")

    for i, case in enumerate(spec.cases, 1):
        print(f"  {i}/{len(spec.cases)}: {case.question}")

        result = pipeline.query(case.question, use_cache=False)

        questions_list.append(case.question)
        answers_list.append(result["answer"])
        contexts_list.append([doc["text"] for doc in result["context_docs"]])
        ground_truths_list.append(case.ground_truth)

        print("     [+] Ответ получен")

    print()

    return Dataset.from_dict(
        {
            "question": questions_list,
            "answer": answers_list,
            "contexts": contexts_list,
            "ground_truth": ground_truths_list,
        }
    )


def _print_metric_block(title: str, result, key: str):
    values = [v for v in result[key] if not (isinstance(v, float) and math.isnan(v))]
    if values:
        avg = sum(values) / len(values)
        print(f"   {title}: {avg:.4f}")
    else:
        print(f"   {title}: нет валидных значений")


def _print_per_question(result, keys, questions):
    for i, question in enumerate(questions):
        print(f"\n{i + 1}. {question}")
        for key in keys:
            label = {
                "faithfulness": "Faithfulness",
                "context_precision": "Context precision (к эталону)",
                "context_utilization": "Context utilization (к ответу RAG)",
            }.get(key, key)
            val = result[key][i]
            if not (isinstance(val, float) and math.isnan(val)):
                print(f"   {label}: {val:.4f}")
            else:
                print(f"   {label}: не удалось вычислить")


def evaluate_rag_system(argv=None):
    args = parse_args(argv)
    load_repo_env()
    apply_privacy_defaults()  # after .env, so an explicit RAGAS_DO_NOT_TRACK there still wins
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ ЧЕРЕЗ RAGAS (опционально, вручную)")
    print("=" * 70)
    print()

    try:
        spec = load_eval_dataset(args.dataset)
        judge = resolve_judge_config(args.judge_model)
    except (EvalDatasetError, EvalConfigError) as e:
        print(f"[ОШИБКА] {e}")
        sys.exit(1)

    print(f"Датасет: {spec.name} ({len(spec.cases)} вопросов) <- {spec.path}")
    print(f"Судья RAGAS: {judge.describe()}")
    print("Вопросы, найденный контекст, ответы и эталоны отправляются этому судье.\n")

    metrics_to_use, run_config = _build_metrics()
    judge_llm, judge_embeddings = build_judge_models(judge)

    try:
        print("[*] Инициализация RAG системы...\n")
        pipeline = RAGPipeline(model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"))
        print(f"\n[OK] RAG система готова к оценке (профиль: {pipeline.profile.name})\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)

    print("=" * 70)
    dataset = prepare_dataset(pipeline, spec)
    print("=" * 70)

    print("\n[*] Запуск оценки метрик RAGAS...")
    if _METRICS_MODE == "v2":
        print("   Метрики: Faithfulness, Context precision (с эталоном), Context utilization (с ответом RAG)")
    else:
        print("   Метрики: Faithfulness, Context precision (legacy-импорт)")
    print("   (несколько минут: вызовы LLM для метрик)\n")

    eval_kw = {"dataset": dataset, "metrics": metrics_to_use, "llm": judge_llm, "embeddings": judge_embeddings}
    if run_config is not None:
        eval_kw["run_config"] = run_config

    try:
        result = evaluate(**eval_kw)
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    keys = _metric_keys(metrics_to_use)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70 + "\n")

    _print_metric_block("Faithfulness (ответ vs контекст)", result, "faithfulness")
    _print_metric_block("Context precision (контекст vs эталон)", result, "context_precision")
    if "context_utilization" in keys:
        _print_metric_block("Context utilization (контекст vs ответ RAG)", result, "context_utilization")

    numeric_avgs = []
    for k in keys:
        vals = [v for v in result[k] if not (isinstance(v, float) and math.isnan(v))]
        if vals:
            numeric_avgs.append(sum(vals) / len(vals))
    avg_score = sum(numeric_avgs) / len(numeric_avgs) if numeric_avgs else 0.0

    print(f"\n{'─' * 70}")
    print(f"[ИТОГО] Среднее по метрикам: {avg_score:.4f}")

    if avg_score >= 0.7:
        print("   Оценка: высокие показатели [OK]")
    elif avg_score >= 0.5:
        print("   Оценка: удовлетворительно [!]")
    else:
        print("   Оценка: есть запас по ретриву/промпту [X]")

    print("\n" + "=" * 70)
    print("ДЕТАЛЬНО ПО ВОПРОСАМ")
    print("=" * 70)
    _print_per_question(result, keys, [case.question for case in spec.cases])

    print("\n" + "=" * 70)
    print("[INFO] КАК ЧИТАТЬ МЕТРИКИ")
    print("=" * 70)
    print("""
Faithfulness — насколько ответ RAG выводим из переданного контекста (без «галлюцинаций»).

Context precision (с эталоном) — насколько каждый извлечённый фрагмент полезен для ответа,
согласующегося с эталоном (ground_truth) из датасета.

Context utilization — то же по смыслу, но опорный «ответ» — фактический ответ вашей RAG;
удобно, когда ответ хороший, а эталон формулирован иначе.

Faithfulness «не удалось вычислить» — чаще всего LLM-судья не разбил ответ на тезисы (пустой список);
повторный запуск или другая модель-судья (--judge-model) может помочь.

Метрики оценивает LLM-судья: это ориентир для сравнения конфигураций, а не доказательство корректности.
    """)

    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()
