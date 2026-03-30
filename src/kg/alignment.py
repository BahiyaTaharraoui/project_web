
import argparse
import time
import csv
import requests
import re
from rdflib import Graph, Literal, RDF, RDFS, OWL, Namespace, URIRef, XSD
from rdflib.namespace import SKOS

# CONFIG
WIKIDATA_API   = "https://www.wikidata.org/w/api.php"
CONFIDENCE_MIN = 0.6      # seuil minimum pour accepter un match
SLEEP_BETWEEN  = 0.3      # secondes entre chaque appel API (politesse)
MAX_ENTITIES   = None     # None = toutes ; mettre ex. 100 pour tester vite

EX  = Namespace("http://myolympicgraph.org/resource/")
ONT = Namespace("http://myolympicgraph.org/ontology/")
WD  = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

TYPE_WIKIDATA_FILTER = {
    ONT.Person:       ["Q5"],               # human
    ONT.Organization: ["Q43229", "Q783794", "Q2659904"],  # org, committee, gov org
    ONT.Place:        ["Q515", "Q6256", "Q35657", "Q82794"],  # city, country, state, region
}

# WIKIDATA 

def search_wikidata(label: str, entity_type_uri, language: str = "en") -> list:
    """
    Recherche une entité sur Wikidata par son label.
    Retourne une liste de candidats triés par pertinence.
    """
    params = {
        "action":      "wbsearchentities",
        "search":      label,
        "language":    language,
        "format":      "json",
        "limit":       5,
        "type":        "item",
    }
    headers = {
        "User-Agent": "OlympicKGProject/1.0 (olypic; bahiyataharraoui70@gmail.com) python-requests"
    }
    try:
        r = requests.get(WIKIDATA_API, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    [WARN] API error for '{label}': {e}")
        return []

    results = []
    for item in data.get("search", []):
        qid         = item.get("id", "")
        item_label  = item.get("label", "")
        description = item.get("description", "")
        score       = _compute_score(label, item_label, description, entity_type_uri)
        results.append({
            "qid":         qid,
            "label":       item_label,
            "description": description,
            "score":       score,
            "uri":         f"http://www.wikidata.org/entity/{qid}",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _compute_score(query: str, candidate_label: str, description: str, entity_type_uri) -> float:

    q = query.lower().strip()
    c = candidate_label.lower().strip()
    d = description.lower()

    if q == c:
        score = 1.0
    elif q in c or c in q:
        score = 0.75
    else:
        common = len(set(q.split()) & set(c.split()))
        total  = max(len(q.split()), 1)
        score  = 0.4 * (common / total)

    type_keywords = {
        ONT.Person:       ["person", "athlete", "politician", "president", "born"],
        ONT.Organization: ["organization", "committee", "federation", "association", "club"],
        ONT.Place:        ["city", "country", "region", "state", "capital", "municipality"],
    }
    for kw in type_keywords.get(entity_type_uri, []):
        if kw in d:
            score = min(score + 0.1, 1.0)
            break

    return round(score, 3)

 #ALIGNEMENT

def align_graph(input_ttl: str, output_ttl: str, mapping_csv: str):

    print("\n=== STEP 2: Entity Linking with Wikidata ===\n")

    g_source = Graph()
    g_source.parse(input_ttl, format="turtle")
    print(f"  Source graph loaded: {len(g_source)} triples")

    g_align = Graph()
    g_align.bind("ex",   EX)
    g_align.bind("ont",  ONT)
    g_align.bind("owl",  OWL)
    g_align.bind("rdfs", RDFS)
    g_align.bind("wd",   WD)
    g_align.bind("skos", SKOS)

    # Récupère toutes les entités avec leur type
    entities = []
    for s, p, o in g_source.triples((None, RDF.type, None)):
        if str(o).startswith(str(ONT)):
            label_triple = list(g_source.triples((s, RDFS.label, None)))
            if label_triple:
                label = str(label_triple[0][2])
                entities.append((s, label, o))

    if MAX_ENTITIES:
        entities = entities[:MAX_ENTITIES]

    print(f"  Entities to align: {len(entities)}\n")

    # Table de mapping
    mapping_rows = []
    matched      = 0
    unmatched    = 0

    for i, (entity_uri, label, entity_type) in enumerate(entities):
        print(f"  [{i+1}/{len(entities)}] '{label}' ({entity_type.split('/')[-1]}) ...", end=" ")

        candidates = search_wikidata(label, entity_type)
        time.sleep(SLEEP_BETWEEN)

        best = candidates[0] if candidates else None

        if best and best["score"] >= CONFIDENCE_MIN:
            wd_uri = URIRef(best["uri"])
            g_align.add((entity_uri, OWL.sameAs,     wd_uri))
            g_align.add((entity_uri, SKOS.exactMatch, wd_uri))
            g_align.add((wd_uri,     RDFS.label,      Literal(best["label"], lang="en")))
            if best["description"]:
                g_align.add((wd_uri, RDFS.comment, Literal(best["description"], lang="en")))

            print(f"  {best['qid']} (score={best['score']})")
            mapping_rows.append({
                "private_entity":  str(entity_uri),
                "label":           label,
                "type":            str(entity_type).split("/")[-1],
                "external_uri":    best["uri"],
                "wikidata_qid":    best["qid"],
                "wikidata_label":  best["label"],
                "description":     best["description"],
                "confidence":      best["score"],
                "status":          "matched",
            })
            matched += 1

        else:
            _define_locally(g_align, entity_uri, label, entity_type)
            score = best["score"] if best else 0.0
            qid   = best["qid"]   if best else ""
            print(f"no match (best score={score})")
            mapping_rows.append({
                "private_entity":  str(entity_uri),
                "label":           label,
                "type":            str(entity_type).split("/")[-1],
                "external_uri":    "",
                "wikidata_qid":    qid,
                "wikidata_label":  best["label"] if best else "",
                "description":     best["description"] if best else "",
                "confidence":      score,
                "status":          "local_only",
            })
            unmatched += 1

    # Sauvegarde CSV
    _save_csv(mapping_rows, mapping_csv)

    # Merge + sérialise
    g_merged = g_source + g_align
    g_merged.serialize(output_ttl, format="turtle")

    print(f"\n{'='*50}")
    print(f" Step 2 complete")
    print(f"   Matched with Wikidata : {matched}")
    print(f"   Local definitions only: {unmatched}")
    print(f"   Alignment triples added: {len(g_align)}")
    print(f"   Total triples (merged) : {len(g_merged)}")
    print(f"   Mapping table → {mapping_csv}")
    print(f"   Alignment graph → {output_ttl}")


def _define_locally(g: Graph, entity_uri: URIRef, label: str, entity_type):

    type_name = str(entity_type).split("/")[-1]

    g.add((entity_uri, RDF.type,      entity_type))
    g.add((entity_uri, RDFS.label,    Literal(label, datatype=XSD.string)))
    g.add((entity_uri, RDFS.comment,  Literal(
        f"Local entity of type {type_name}, not found in external KBs.",
        lang="en"
    )))
    # Marque explicitement comme entité locale
    g.add((entity_uri, ONT.isLocalEntity, Literal(True, datatype=XSD.boolean)))


def _save_csv(rows: list, path: str):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Mapping CSV saved → {path}")


#FORMATTER

def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: Entity Linking with Wikidata")
    parser.add_argument("--graph",  default="././kg_artifacts/graph.ttl",
                        help="Input TTL from Step 1")
    parser.add_argument("--out",    default="././kg_artifacts/alignment.ttl",
                        help="Output TTL with alignment triples")
    parser.add_argument("--csv",    default="././kg_artifacts/mapping_table.csv",
                        help="Output mapping table")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Limit number of entities to align")
    return parser.parse_args()


if __name__ == "__main__":
    import os
    os.makedirs("././kg_artifacts", exist_ok=True)

    args = parse_args()
    if args.limit:
        MAX_ENTITIES = args.limit

    align_graph(
        input_ttl   = args.graph,
        output_ttl  = args.out,
        mapping_csv = args.csv,
    )