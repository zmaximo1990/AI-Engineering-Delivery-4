# Load env before the modules that read it at import time.
from pathlib import Path

from ingestion import Ingestor
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

import argparse
import asyncio
import logging

from chain import get_rag_response
from samples import DEMO_QUESTIONS
from schemas import LLMConfig, LLMProvider, RAGAnswer
from tests.evaluate import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question against the indexed documents, or score retrieval."
    )
    parser.add_argument(
        "-p",
        "--provider",
        default=LLMProvider.OPENAI.value,
        choices=[provider.value for provider in LLMProvider],
        help="LLM provider used to build LLMConfig (default: openai)",
    )
    parser.add_argument(
        "-q",
        "--question",
        default=None,
        help="Ask one question and exit. If omitted, prompt for questions.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the grounded question and the trap question, then exit.",
    )
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Score the golden set and the unanswerable set, then exit.",
    )
    args = parser.parse_args()
    selected = [
        name
        for name, enabled in (
            ("--demo", args.demo),
            ("--question", bool(args.question)),
            ("--eval", args.eval),
        )
        if enabled
    ]
    if len(selected) > 1:
        parser.error("use only one of --demo, --question, or --eval")
    return args

def prepare_index() -> int:
    """Open the local store and index ``data/`` when the collection is empty.

    Called before the prompt so the embedding model is loaded once, up front.
    """
    logger.info("Preparing index")
    Ingestor().ingest_if_needed()

def print_response(result: RAGAnswer) -> None:
    print(f"Answer: {result.answer}", flush=True)
    if result.sources:
        print("Sources: " + ", ".join(result.sources), flush=True)
    else:
        print("Sources: (none)", flush=True)
    print(f"Retrieved fragments: {result.retrieved_fragments}", flush=True)


async def ask(config: LLMConfig, question: str) -> None:
    if not question or not question.strip():
        logger.error("Question is empty")
        return
    result = await get_rag_response(question, config)
    print_response(result)


def read_question() -> str:
    try:
        return input("Question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


async def main(args: argparse.Namespace) -> None:
    config = LLMConfig(provider=LLMProvider(args.provider))
    logger.info("LLMConfig initialized (provider=%s)", config.provider.value)
    prepare_index()

    if args.eval:
        await run_evaluation(config)
        logger.info("Finished")
        return

    if args.demo:
        for label, question in DEMO_QUESTIONS:
            print(f"\n--- {label} ---", flush=True)
            print(f"Question: {question}", flush=True)
            try:
                await ask(config, question)
            except Exception as exc:
                logger.error("get_rag_response failed for %s: %s", label, exc)
        logger.info("Finished")
        return

    if args.question:
        await ask(config, args.question)
        logger.info("Finished")
        return

    print("TechCorp policy assistant... ask away!\n\n", flush=True)
    print("Type a question and press Enter. An empty line, exit, or quit stops the prompt.", flush=True)
    while True:
        question = read_question()
        if not question or question.lower() in {"exit", "quit"}:
            break
        try:
            await ask(config, question)
        except Exception as exc:
            logger.error("get_rag_response failed: %s", exc)
    logger.info("Finished")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
