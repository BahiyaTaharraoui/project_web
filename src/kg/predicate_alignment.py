
import argparse
import time
import csv
import requests
from rdflib import Graph, Literal, RDF, RDFS, OWL, Namespace, URIRef, XSD

# CONFIG
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
SLEEP_BETWEEN   = 1.0   # secondes entre requêtes SPARQL (politesse)

HEADERS = {
    "User-Agent":  "OlympicKGProject/1.0 (student-project) python-requests/2.x",
    "Accept":      "application/sparql-results+json",
}

EX  = Namespace("http://myolympicgraph.org/resource/")
ONT = Namespace("http://myolympicgraph.org/ontology/")
WD  = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

#PRIVATE PREDICATES TO ALIGN

PRIVATE_PREDICATES = {
    ONT.memberOf: {
        "label":       "member of",
        "keywords":    ["member", "member of", "affiliated"],
        "description": "A person is affiliated with or member of an organization",
        "expected_relation": "equivalent",   # owl:equivalentProperty
    },
    ONT.basedIn: {
        "label":       "based in",
        "keywords":    ["country of citizenship", "place of birth", "residence", "nationality"],
        "description": "A person is associated with a geopolitical location",
        "expected_relation": "broader",      # rdfs:subPropertyOf (plus spécifique)
    },
    ONT.locatedIn: {
        "label":       "located in",
        "keywords":    ["located in", "headquarters location", "country"],
        "description": "An organization is located in a place",
        "expected_relation": "equivalent",
    },
    ONT.mentionedIn: {
        "label":       "mentioned in",
        "keywords":    ["described by source", "reference", "sourced from"],
        "description": "Source page where the entity was extracted",
        "expected_relation": "broader",
    },
}

# SPARQL QUERY Wikidata

SPARQL_BY_LABEL = """
SELECT ?property ?propertyLabel ?propertyDescription WHERE {{
  ?property a wikibase:Property .
  ?property rdfs:label ?propertyLabel .
  FILTER(LANG(?propertyLabel) = "en")
  FILTER(CONTAINS(LCASE(?propertyLabel), "{keyword}"))
  OPTIONAL {{ ?property schema:description ?propertyDescription .
              FILTER(LANG(?propertyDescription) = "en") }}
}}
LIMIT 10
"""

SPARQL_BY_SUBJECT_OBJECT = """
SELECT DISTINCT ?property ?propertyLabel WHERE {{
  {subject_wd} ?property {object_wd} .
  ?prop wikibase:directClaim ?property .
  ?prop rdfs:label ?propertyLabel .
  FILTER(LANG(?propertyLabel) = "en")
}}
LIMIT 10
"""


def sparql_search_by_label(keyword: str) -> list:
 
    query = SPARQL_BY_LABEL.format(keyword=keyword.lower())
    try:
        r = requests.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        bindings = r.json().get("results", {}).get("bindings", [])
        results = []
        for b in bindings:
            uri   = b.get("property", {}).get("value", "")
            label = b.get("propertyLabel", {}).get("value", "")
            desc  = b.get("propertyDescription", {}).get("value", "")
            # Extrait le PID (ex: P463 depuis l'URI)
            pid   = uri.split("/")[-1] if "/" in uri else uri
            if pid.startswith("P"):
                results.append({"pid": pid, "uri": uri, "label": label, "description": desc})
        return results
    except Exception as e:
        print(f"    [WARN] SPARQL error for '{keyword}': {e}")
        return []


def sparql_search_by_subject_object(subject_wd_uri: str, object_wd_uri: str) -> list:

    query = SPARQL_BY_SUBJECT_OBJECT.format(
        subject_wd=f"<{subject_wd_uri}>",
        object_wd=f"<{object_wd_uri}>",
    )
    try:
        r = requests.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        bindings = r.json().get("results", {}).get("bindings", [])
        return [
            {
                "pid":   b.get("property", {}).get("value", "").split("/")[-1],
                "uri":   b.get("property", {}).get("value", ""),
                "label": b.get("propertyLabel", {}).get("value", ""),
            }
            for b in bindings
        ]
    except Exception as e:
        print(f"    [WARN] Subject-Object SPARQL error: {e}")
        return []


# SELECTION DU MEILLEUR CANDIDAT

# Mapping manuel validée
MANUAL_VALIDATED = {
    ONT.memberOf: {
        "pid":         "P463",
        "uri":         "http://www.wikidata.org/prop/direct/P463",
        "label":       "member of",
        "description": "organization, club, political party, etc. of which the subject is a member",
        "relation":    "equivalent",
        "confidence":  1.0,
    },
    ONT.basedIn: {
        "pid":         "P27",
        "uri":         "http://www.wikidata.org/prop/direct/P27",
        "label":       "country of citizenship",
        "description": "the object is a country that recognizes the subject as its citizen",
        "relation":    "broader",   # basedIn est plus général que country of citizenship
        "confidence":  0.8,
    },
    ONT.locatedIn: {
        "pid":         "P131",
        "uri":         "http://www.wikidata.org/prop/direct/P131",
        "label":       "located in the administrative territorial entity",
        "description": "the item is located on the territory of this administrative entity",
        "relation":    "equivalent",
        "confidence":  0.85,
    },
    ONT.mentionedIn: {
        "pid":         "P1343",
        "uri":         "http://www.wikidata.org/prop/direct/P1343",
        "label":       "described by source",
        "description": "the subject is described by the following source",
        "relation":    "broader",
        "confidence":  0.75,
    },
}


def select_best_candidate(candidates: list, predicate_info: dict) -> dict | None:
    """
    Sélectionne le meilleur candidat parmi les résultats SPARQL.
    Priorité : correspondance exacte du label → correspondance partielle.
    """
    target_label = predicate_info["label"].lower()
    keywords     = [k.lower() for k in predicate_info["keywords"]]

    exact_match   = None
    partial_match = None

    for c in candidates:
        c_label = c.get("label", "").lower()
        if c_label == target_label:
            exact_match = c
            break
        for kw in keywords:
            if kw in c_label or c_label in kw:
                if partial_match is None:
                    partial_match = c

    return exact_match or partial_match


# ALIGNMENT PIPELINE


def align_predicates(input_ttl: str, output_ttl: str, mapping_csv: str):
    """Pipeline complet d'alignement des prédicats."""

    print("\n=== STEP 3: Predicate Alignment with Wikidata ===\n")

    # Charge le graphe
    g = Graph()
    g.parse(input_ttl, format="turtle")
    g.bind("ex",  EX)
    g.bind("ont", ONT)
    g.bind("owl", OWL)
    g.bind("wdt", WDT)
    g.bind("wd",  WD)
    print(f"  Graph loaded: {len(g)} triples\n")

    mapping_rows = []

    for pred_uri, pred_info in PRIVATE_PREDICATES.items():
        pred_label = pred_info["label"]
        print(f"  ── Aligning predicate: <{pred_uri.split('/')[-1]}> ('{pred_label}')")

        # 1. Cherche via SPARQL pour chaque keyword
        all_candidates = []
        for kw in pred_info["keywords"]:
            print(f"     SPARQL search: keyword='{kw}' ...", end=" ")
            results = sparql_search_by_label(kw)
            print(f"{len(results)} candidates found")
            all_candidates.extend(results)
            time.sleep(SLEEP_BETWEEN)

        # Dédoublonne par PID
        seen_pids  = set()
        candidates = []
        for c in all_candidates:
            if c["pid"] not in seen_pids:
                seen_pids.add(c["pid"])
                candidates.append(c)

        # 2. Sélection automatique du meilleur candidat SPARQL
        sparql_best = select_best_candidate(candidates, pred_info)

        # 3. Priorité à la validation manuelle 
        manual = MANUAL_VALIDATED.get(pred_uri)
        best   = manual if manual else (
            {
                "pid":         sparql_best["pid"],
                "uri":         sparql_best["uri"],
                "label":       sparql_best["label"],
                "description": sparql_best.get("description", ""),
                "relation":    pred_info["expected_relation"],
                "confidence":  0.7,
            } if sparql_best else None
        )

        if best:
            wd_prop_uri = URIRef(best["uri"])

            # Ajoute l'alignement selon le type de relation
            if best["relation"] == "equivalent":
                g.add((pred_uri, OWL.equivalentProperty, wd_prop_uri))
                relation_type = "owl:equivalentProperty"
                print(f"     Equivalent → wdt:{best['pid']} \"{best['label']}\" (conf={best['confidence']})")
            else:
                # "broader" → notre prédicat est subPropertyOf du Wikidata
                g.add((pred_uri, RDFS.subPropertyOf, wd_prop_uri))
                relation_type = "rdfs:subPropertyOf"
                print(f"      SubProperty → wdt:{best['pid']} \"{best['label']}\" (conf={best['confidence']})")

            # Documente le prédicat Wikidata dans le graphe
            g.add((wd_prop_uri, RDFS.label,   Literal(best["label"],       lang="en")))
            g.add((wd_prop_uri, RDFS.comment, Literal(best["description"], lang="en")))

            mapping_rows.append({
                "private_predicate":    str(pred_uri),
                "private_label":        pred_label,
                "wikidata_pid":         best["pid"],
                "wikidata_uri":         best["uri"],
                "wikidata_label":       best["label"],
                "wikidata_description": best["description"],
                "relation_type":        relation_type,
                "confidence":           best["confidence"],
                "validation":           "manual" if manual else "sparql_auto",
            })

        else:
            print(f"     NO MATCH found — keeping local definition only")
            mapping_rows.append({
                "private_predicate":    str(pred_uri),
                "private_label":        pred_label,
                "wikidata_pid":         "",
                "wikidata_uri":         "",
                "wikidata_label":       "",
                "wikidata_description": "",
                "relation_type":        "none",
                "confidence":           0.0,
                "validation":           "unmatched",
            })

        print()

    # Sauvegarde CSV
    _save_csv(mapping_rows, mapping_csv)

    # Sauvegarde TTL
    g.serialize(output_ttl, format="turtle")

    # Résumé
    matched = sum(1 for r in mapping_rows if r["wikidata_pid"])
    print(f"{'='*55}")
    print(f"   Step 3 complete")
    print(f"   Predicates aligned : {matched} / {len(PRIVATE_PREDICATES)}")
    print(f"   Total triples      : {len(g)}")
    print(f"   Mapping CSV        → {mapping_csv}")
    print(f"   Output TTL         → {output_ttl}")
    print()
    _print_summary(mapping_rows)


def _print_summary(rows: list):
    """Affiche un tableau récapitulatif dans le terminal."""
    print("\n  Predicate Alignment Summary:")
    print(f"  {'Private predicate':<15} {'Wikidata PID':<14} {'Relation':<25} {'Conf'}")
    print("  " + "-"*65)
    for r in rows:
        pred  = r["private_label"]
        pid   = r["wikidata_pid"] or "—"
        rel   = r["relation_type"].split(":")[-1] if r["relation_type"] != "none" else "—"
        conf  = r["confidence"]
        print(f"  {pred:<15} {pid:<14} {rel:<25} {conf}")


def _save_csv(rows: list, path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Mapping CSV saved → {path}")


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Step 3: Predicate Alignment with Wikidata")
    parser.add_argument("--graph", default="././kg_artifacts/alignment.ttl",
                        help="Input TTL from Step 2")
    parser.add_argument("--out",   default="././kg_artifacts/predicate_alignment.ttl",
                        help="Output TTL with predicate alignment triples")
    parser.add_argument("--csv",   default="././kg_artifacts/predicate_mapping.csv",
                        help="Output predicate mapping table (CSV)")
    return parser.parse_args()


if __name__ == "__main__":
    import os
    os.makedirs("././kg_artifacts", exist_ok=True)
    args = parse_args()
    align_predicates(
        input_ttl   = args.graph,
        output_ttl  = args.out,
        mapping_csv = args.csv,
    )