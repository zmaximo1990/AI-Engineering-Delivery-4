"""Score retrieval on the golden set and abstention on the unanswerable set.

Precision@k is the share of the top-k chunks that come from an expected
document. Recall@k is the share of expected documents that appear in those
chunks. A question with one expected document therefore has Recall@k of 1
when that document is in the top k, and 0 otherwise.

MRR@k is the mean reciprocal rank. For one question the reciprocal rank is
1 divided by the position of the first relevant chunk in the top k, or 0
when none of those chunks comes from an expected document. Position 1 gives
1.00, position 2 gives 0.50, and a miss gives 0.

Unanswerable questions must be declined. The answer has to say that the
information is not in the documentation, and it must not match a
hallucination pattern from the set.

Golden cases cite files in ``data/``. Each cited file must exist and its
name must start with the document id.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import argparse
import asyncio
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document

from chain import get_rag_response
from ingestion import Ingestor
from retriever import Retriever
from schemas import LLMConfig, LLMProvider
from settings import DOC_ID, settings

logger = logging.getLogger(__name__)

_TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = _TESTS_DIR / "golden_set.json"
UNANSWERABLE_PATH = _TESTS_DIR / "unanswerable_set.json"

# The prompt tells the model to refuse, in the language of the question,
# when the context does not contain the answer.
_ABSTENTION_PATTERNS = (
    r"no tengo acceso",
    r"do not have access",
    r"don't have access",
    r"documentos disponibles",
    r"documentaci[oó]n disponible",
    r"available documents",
    r"no (?:hay|existe|cuento con|dispongo de) (?:esa |la )?informaci[oó]n",
    r"informaci[oó]n no (?:est[aá]|figura|aparece|consta|se encuentra)",
    r"no est[aá] en (?:el |la |los )?(?:contexto|document)",
    r"no (?:se )?(?:encuentra|figura|aparece|menciona) en (?:el |la |los )?(?:contexto|document)",
    r"not in the (?:provided |available )?(?:context|documents|documentation)",
    r"no (?:est[aá] |se encuentra )?(?:documentad[oa]|especificad[oa])",
)
_ABSTENTION = re.compile(
    "|".join(f"(?:{pattern})" for pattern in _ABSTENTION_PATTERNS),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GoldenResult:
    """Precision@k, Recall@k, and reciprocal rank for one golden question."""

    case_id: str
    useful: int
    found: int
    expected_count: int
    k: int
    retrieved: tuple[str, ...]
    expected: tuple[str, ...]
    first_rank: int | None = None
    error: str | None = None

    @property
    def precision(self) -> float:
        if self.error is not None or self.k <= 0:
            return 0.0
        return self.useful / self.k

    @property
    def recall(self) -> float:
        if self.error is not None or self.expected_count <= 0:
            return 0.0
        return self.found / self.expected_count

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first relevant chunk in the top k, or 0 on a miss."""
        if self.error is not None or self.first_rank is None or self.first_rank < 1:
            return 0.0
        return 1.0 / self.first_rank


@dataclass(frozen=True)
class UnanswerableResult:
    """Abstention and hallucination outcome for one unanswerable question."""

    case_id: str
    abstained: bool
    hallucinated: bool
    answer: str
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.abstained and not self.hallucinated


def load_cases(path: Path) -> list[dict]:
    """Load a JSON list of evaluation cases."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path.name} must contain a non-empty list of cases")
    for index, case in enumerate(payload, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"{path.name}: case {index} is not an object")
    return payload


def document_id(document: Document) -> str:
    """Return the NE- document id stored on a chunk, or parsed from its file name."""
    raw = document.metadata.get("doc_id")
    if raw:
        return str(raw).strip().upper()
    source = str(document.metadata.get("source", ""))
    match = DOC_ID.match(source)
    if match:
        return match.group(1).upper()
    return source


def abstains(answer: str) -> bool:
    """Return whether the answer says the documentation does not contain the fact."""
    return _ABSTENTION.search(answer) is not None


def hallucination_pattern(answer: str, patterns: list[str]) -> str | None:
    """Return the first hallucination pattern that matches, or None."""
    for pattern in patterns:
        try:
            if re.search(pattern, answer, re.IGNORECASE):
                return pattern
        except re.error:
            logger.warning("Skipping invalid hallucination pattern %r", pattern)
    return None


def _case_id(case: dict, fallback: str) -> str:
    value = case.get("id", fallback)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{fallback}: id must be a non-empty string")
    return value.strip()


def _question(case: dict, case_id: str) -> str:
    question = case.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{case_id}: question must be a non-empty string")
    return question.strip()


def _string_list(value: object, case_id: str, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{case_id}: {field} must be a list of strings")
    cleaned = [item.strip() for item in value if item.strip()]
    if len(cleaned) != len(value):
        raise ValueError(f"{case_id}: {field} contains an empty string")
    return cleaned


def _unique(items: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _expected(case: dict, case_id: str) -> tuple[list[str], set[str], dict[str, str]]:
    expected_ids = _unique(
        item.upper()
        for item in _string_list(case.get("relevant_docs"), case_id, "relevant_docs")
    )
    if not expected_ids:
        raise ValueError(f"{case_id}: relevant_docs is empty")
    expected_files = _string_list(
        case.get("relevant_files") or [],
        case_id,
        "relevant_files",
    )
    file_to_id = {
        filename: doc_id
        for doc_id, filename in zip(expected_ids, expected_files)
    }
    return expected_ids, set(expected_files), file_to_id


def score_golden_case(
    retriever: Retriever,
    case: dict,
    k: int,
    fallback_id: str,
) -> GoldenResult:
    """Retrieve the top k chunks and score them against the expected documents."""
    case_id = _case_id(case, fallback_id)
    expected_ids, expected_files, file_to_id = _expected(case, case_id)
    question = _question(case, case_id)
    try:
        documents = list(retriever.invoke(question))[:k]
    except Exception as exc:
        logger.exception("Retrieval failed for %s", case_id)
        return GoldenResult(
            case_id=case_id,
            useful=0,
            found=0,
            expected_count=len(expected_ids),
            k=k,
            retrieved=(),
            expected=tuple(expected_ids),
            error=str(exc),
        )

    expected_id_set = set(expected_ids)
    useful = 0
    found: set[str] = set()
    retrieved: list[str] = []
    first_rank: int | None = None
    for rank, document in enumerate(documents, start=1):
        doc_id = document_id(document)
        source = str(document.metadata.get("source", ""))
        if doc_id and doc_id not in retrieved:
            retrieved.append(doc_id)
        matched_id = doc_id in expected_id_set
        matched_file = source in expected_files
        if not matched_id and not matched_file:
            continue
        useful += 1
        if first_rank is None:
            first_rank = rank
        if matched_id:
            found.add(doc_id)
        mapped = file_to_id.get(source)
        if mapped is not None and mapped in expected_id_set:
            found.add(mapped)

    return GoldenResult(
        case_id=case_id,
        useful=useful,
        found=len(found),
        expected_count=len(expected_ids),
        k=k,
        retrieved=tuple(retrieved),
        expected=tuple(expected_ids),
        first_rank=first_rank,
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _join(items: tuple[str, ...] | list[str]) -> str:
    return ",".join(items) if items else "-"


def _print_golden_header(total: int, k: int) -> None:
    print(f"\nGolden set ({total} questions, k={k})", flush=True)
    print(
        f"Precision@{k} = chunks from an expected document / {k}. "
        f"Recall@{k} = expected documents present in the top {k} / expected documents. "
        f"RR@{k} = 1 / rank of the first relevant chunk (0 if none). "
        f"MRR@{k} is the mean of those reciprocal ranks.",
        flush=True,
    )


def _print_golden_row(row: GoldenResult, k: int) -> None:
    if row.error:
        print(f"{row.case_id:<6} ERROR  {row.error}", flush=True)
        return
    rank = "-" if row.first_rank is None else str(row.first_rank)
    print(
        f"{row.case_id:<6} "
        f"P@{k}={row.precision:.2f} ({row.useful}/{k})  "
        f"R@{k}={row.recall:.2f} ({row.found}/{row.expected_count})  "
        f"RR@{k}={row.reciprocal_rank:.2f} (rank {rank})  "
        f"retrieved={_join(row.retrieved)}  expected={_join(row.expected)}",
        flush=True,
    )


def _print_golden_summary(rows: list[GoldenResult], k: int) -> None:
    scored = [row for row in rows if row.error is None]
    hits = sum(1 for row in scored if row.recall > 0)
    print(flush=True)
    if not scored:
        print(f"Mean Precision@{k}: n/a", flush=True)
        print(f"Mean Recall@{k}: n/a", flush=True)
        print(f"MRR@{k}: n/a", flush=True)
        print(f"Hit@{k}: 0/{len(rows)}", flush=True)
        return
    print(f"Mean Precision@{k}: {_mean([row.precision for row in scored]):.2f}", flush=True)
    print(f"Mean Recall@{k}: {_mean([row.recall for row in scored]):.2f}", flush=True)
    print(f"MRR@{k}: {_mean([row.reciprocal_rank for row in scored]):.2f}", flush=True)
    print(
        f"Hit@{k}: {hits}/{len(scored)} questions retrieved at least one expected document",
        flush=True,
    )
    if len(scored) != len(rows):
        print(f"Completed without error: {len(scored)}/{len(rows)}", flush=True)


def _excerpt(answer: str) -> str:
    text = " ".join(answer.split())
    if len(text) <= 180:
        return text
    return text[:177] + "..."


def _print_unanswerable(rows: list[UnanswerableResult], total: int) -> None:
    print(f"\nUnanswerable set ({total} questions)", flush=True)
    print(
        "Pass = the answer says the information is not in the documentation, "
        "and it matches no hallucination pattern.",
        flush=True,
    )
    for row in rows:
        if row.error:
            print(f"{row.case_id:<6} ERROR  {row.error}", flush=True)
            continue
        abstained = "yes" if row.abstained else "no"
        hallucinated = "yes" if row.hallucinated else "no"
        status = "pass" if row.passed else "fail"
        print(
            f"{row.case_id:<6} abstained={abstained:<3}  "
            f"hallucination={hallucinated:<3}  {status}",
            flush=True,
        )
        if not row.passed and row.answer:
            print(f"       {_excerpt(row.answer)}", flush=True)

    scored = [row for row in rows if row.error is None]
    passed = sum(1 for row in scored if row.passed)
    abstained = sum(1 for row in scored if row.abstained)
    hallucinated = sum(1 for row in scored if row.hallucinated)
    print(flush=True)
    if scored:
        print(f"Abstained: {abstained}/{len(scored)}", flush=True)
        print(f"Hallucination signals: {hallucinated}/{len(scored)}", flush=True)
        print(f"Passed: {passed}/{len(scored)}", flush=True)
    else:
        print("Passed: 0/0", flush=True)
    if len(scored) != total:
        print(f"Completed without error: {len(scored)}/{total}", flush=True)
    if len(rows) < total and rows and rows[-1].error and "is not configured" in rows[-1].error:
        print("Stopped because the provider key is missing.", flush=True)


def _quiet_logs() -> None:
    logging.getLogger().setLevel(logging.WARNING)


def _validate_golden(cases: list[dict]) -> None:
    """Check questions and that each cited file exists for its document id."""
    for index, case in enumerate(cases, start=1):
        case_id = _case_id(case, f"golden[{index}]")
        _question(case, case_id)
        _expected(case, case_id)
        doc_ids = _string_list(case.get("relevant_docs"), case_id, "relevant_docs")
        filenames = _string_list(
            case.get("relevant_files") or [],
            case_id,
            "relevant_files",
        )
        if len(filenames) != len(doc_ids):
            raise ValueError(
                f"{case_id}: relevant_files must list one file per relevant_docs entry"
            )
        for doc_id, filename in zip(doc_ids, filenames):
            path = settings.data_dir / filename
            if not path.is_file():
                raise ValueError(f"{case_id}: data file not found: {filename}")
            match = DOC_ID.match(filename)
            if match is None or match.group(1).upper() != doc_id.upper():
                raise ValueError(f"{case_id}: {filename} is not the file for {doc_id}")


def _validate_unanswerable(cases: list[dict]) -> None:
    for index, case in enumerate(cases, start=1):
        case_id = _case_id(case, f"unanswerable[{index}]")
        _question(case, case_id)
        signals = case.get("hallucination_signals") or []
        _string_list(signals, case_id, "hallucination_signals")


async def _score_unanswerable(
    cases: list[dict],
    config: LLMConfig | None,
) -> list[UnanswerableResult]:
    rows: list[UnanswerableResult] = []
    for index, case in enumerate(cases, start=1):
        case_id = _case_id(case, f"unanswerable[{index}]")
        question = _question(case, case_id)
        signals = _string_list(
            case.get("hallucination_signals") or [],
            case_id,
            "hallucination_signals",
        )
        try:
            result = await get_rag_response(question, config)
        except ValueError as exc:
            logger.error("Unanswerable question %s failed: %s", case_id, exc)
            rows.append(
                UnanswerableResult(
                    case_id=case_id,
                    abstained=False,
                    hallucinated=False,
                    answer="",
                    error=str(exc),
                )
            )
            if "is not configured" in str(exc):
                break
            continue
        except Exception as exc:
            logger.exception("Unanswerable question %s failed", case_id)
            rows.append(
                UnanswerableResult(
                    case_id=case_id,
                    abstained=False,
                    hallucinated=False,
                    answer="",
                    error=str(exc),
                )
            )
            continue

        answer = result.answer
        rows.append(
            UnanswerableResult(
                case_id=case_id,
                abstained=abstains(answer),
                hallucinated=hallucination_pattern(answer, signals) is not None,
                answer=answer,
            )
        )
    return rows


async def run_evaluation(config: LLMConfig | None = None, k: int | None = None) -> None:
    """Score both sets and print the console report.

    The index must already exist. ``main.py`` prepares it before calling this.
    ``k`` defaults to ``settings.hybrid_k`` (5).
    """
    cutoff = settings.hybrid_k if k is None else k
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError("k must be a positive integer.")

    golden_cases = load_cases(GOLDEN_PATH)
    unanswerable_cases = load_cases(UNANSWERABLE_PATH)
    _validate_golden(golden_cases)
    _validate_unanswerable(unanswerable_cases)

    print(
        f"Scoring {len(golden_cases)} golden questions and "
        f"{len(unanswerable_cases)} unanswerable questions (k={cutoff}).",
        flush=True,
    )
    print("Loading the hybrid retriever...", flush=True)
    _quiet_logs()
    retriever = Retriever(k=cutoff)

    _print_golden_header(len(golden_cases), cutoff)
    golden_rows: list[GoldenResult] = []
    for index, case in enumerate(golden_cases, start=1):
        row = score_golden_case(retriever, case, cutoff, f"golden[{index}]")
        golden_rows.append(row)
        _print_golden_row(row, cutoff)
    _print_golden_summary(golden_rows, cutoff)

    print(
        f"\nScoring {len(unanswerable_cases)} unanswerable questions with the RAG chain.",
        flush=True,
    )
    unanswerable_rows = await _score_unanswerable(unanswerable_cases, config)
    _print_unanswerable(unanswerable_rows, len(unanswerable_cases))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Precision@k, Recall@k, MRR@k, and abstention on the evaluation sets."
    )
    parser.add_argument(
        "-p",
        "--provider",
        default=None,
        choices=[provider.value for provider in LLMProvider],
        help="LLM provider for the unanswerable set (default: LLM_PROVIDER or openai).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        help="Chunks kept per question (default: hybrid_k, 5).",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare the index and print the evaluation report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    Ingestor().ingest_if_needed()
    config = None
    if args.provider:
        config = LLMConfig(provider=LLMProvider(args.provider))
    asyncio.run(run_evaluation(config, k=args.k))


if __name__ == "__main__":
    main()
