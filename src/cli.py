from __future__ import annotations

import argparse

from .rag_pipeline import ClinicalRAG


def _add_ingest_parser(subparsers: argparse._SubParsersAction) -> None:
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest guideline text files into Chroma.",
    )
    ingest_parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing vectors before indexing.",
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=220,
        help="Chunk size in words.",
    )
    ingest_parser.add_argument(
        "--overlap",
        type=int,
        default=40,
        help="Chunk overlap in words.",
    )


def _add_query_parser(subparsers: argparse._SubParsersAction) -> None:
    query_parser = subparsers.add_parser(
        "query",
        help="Run a one-off question against indexed docs.",
    )
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clinical RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_ingest_parser(subparsers)
    _add_query_parser(subparsers)

    return parser.parse_args()


def run() -> None:
    args = parse_args()
    rag = ClinicalRAG()

    if args.command == "ingest":
        stats = rag.ingest(
            clear_existing=args.clear,
            chunk_size_words=args.chunk_size,
            overlap_words=args.overlap,
        )
        print("Ingestion complete")
        print(f"- Files processed: {stats['files']}")
        print(f"- Chunks indexed: {stats['chunks']}")
        print(f"- Collection size: {rag.collection.count()}")
        return

    if args.command == "query":
        answer, sources = rag.answer_question(question=args.question, top_k=args.top_k)
        print("\nAnswer:\n")
        print(answer)
        print("\nSources:\n")
        if not sources:
            print("No sources found")
            return
        for idx, source in enumerate(sources, start=1):
            print(
                f"{idx}. source={source.source}, chunk={source.chunk_index}, distance={source.score:.4f}"
            )


if __name__ == "__main__":
    run()
