"""
src/kg/stats.py
===============
Print statistics about the knowledge graph.

Usage:
    python src/kg/stats.py --graph kg_artifacts/expanded.nt
"""

import argparse
from collections import Counter

from rdflib import Graph, RDF, RDFS


def stats(graph_path: str) -> None:
    fmt = "nt" if graph_path.endswith(".nt") else "turtle"
    g = Graph()
    g.parse(graph_path, format=fmt)

    n_triples  = len(g)
    n_subjects = len(set(g.subjects()))
    n_preds    = len(set(g.predicates()))
    n_objects  = len(set(g.objects()))

    # Class distribution
    class_counter: Counter = Counter()
    for _, _, cls in g.triples((None, RDF.type, None)):
        class_counter[str(cls)] += 1

    # Predicate distribution
    pred_counter: Counter = Counter()
    for _, p, _ in g:
        pred_counter[str(p)] += 1

    print("=" * 60)
    print(f"  Knowledge Base Statistics — {graph_path}")
    print("=" * 60)
    print(f"  Triples   : {n_triples:,}")
    print(f"  Subjects  : {n_subjects:,}")
    print(f"  Predicates: {n_preds:,}")
    print(f"  Objects   : {n_objects:,}")

    print("\n  Top 10 classes:")
    for cls, cnt in class_counter.most_common(10):
        short = cls.split("/")[-1].split("#")[-1]
        print(f"    {short:<30} {cnt:>6}")

    print("\n  Top 10 predicates:")
    for pred, cnt in pred_counter.most_common(10):
        short = pred.split("/")[-1].split("#")[-1]
        print(f"    {short:<30} {cnt:>6}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="KG statistics")
    parser.add_argument("--graph", default="kg_artifacts/expanded.nt")
    args = parser.parse_args()
    stats(args.graph)


if __name__ == "__main__":
    main()
