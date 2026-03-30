
import argparse
import time
import requests
from collections import defaultdict
from rdflib import Graph, Literal, RDF, RDFS, OWL, Namespace, URIRef, XSD

# CONFIG

WIKIDATA_SPARQL  = "https://query.wikidata.org/sparql"
SLEEP_BETWEEN    = 1.0        # secondes entre requêtes (politesse)
DEFAULT_LIMIT    = 500        # triplets max par requête SPARQL
MAX_ENTITIES     = 200        # entités alignées max à expander (1-hop)
MAX_2HOP         = 100        # entités max pour le 2-hop
TARGET_MIN       = 50_000     # objectif minimum de triplets
TARGET_MAX       = 200_000    # objectif maximum

# Prédicats Wikidata à exclure car trop "literal-heavy" ou peu informatifs
BLACKLIST_PREDICATES = {
    "http://www.wikidata.org/prop/direct/P18",    # image
    "http://www.wikidata.org/prop/direct/P94",    # coat of arms image
    "http://www.wikidata.org/prop/direct/P154",   # logo image
    "http://www.wikidata.org/prop/direct/P856",   # official website
    "http://www.wikidata.org/prop/direct/P373",   # Commons category
    "http://www.wikidata.org/prop/direct/P935",   # Commons gallery
    "http://www.wikidata.org/prop/direct/P948",   # page banner
    "http://www.wikidata.org/prop/direct/P1566",  # GeoNames ID
    "http://www.wikidata.org/prop/direct/P214",   # VIAF ID
    "http://www.wikidata.org/prop/direct/P244",   # LC authority ID
    "http://www.wikidata.org/prop/direct/P227",   # GND ID
    "http://www.wikidata.org/prop/direct/P268",   # BnF ID
    "http://www.wikidata.org/prop/direct/P269",   # SUDOC ID
}

# Prédicats Wikidata prioritaires pour l'expansion (sémantiquement riches)
PRIORITY_PREDICATES = [
    ("P463",  "member of"),
    ("P131",  "located in administrative territorial entity"),
    ("P27",   "country of citizenship"),
    ("P106",  "occupation"),
    ("P21",   "sex or gender"),
    ("P569",  "date of birth"),
    ("P570",  "date of death"),
    ("P19",   "place of birth"),
    ("P20",   "place of death"),
    ("P166",  "award received"),
    ("P17",   "country"),
    ("P31",   "instance of"),
    ("P279",  "subclass of"),
    ("P361",  "part of"),
    ("P155",  "follows"),
    ("P156",  "followed by"),
    ("P585",  "point in time"),
    ("P571",  "inception"),
    ("P576",  "dissolved"),
    ("P112",  "founded by"),
    ("P749",  "parent organization"),
    ("P159",  "headquarters location"),
    ("P495",  "country of origin"),
    ("P577",  "publication date"),
]

HEADERS = {
    "User-Agent": "OlympicKGProject/1.0 (student-project) python-requests/2.x",
    "Accept":     "application/sparql-results+json",
}

EX  = Namespace("http://myolympicgraph.org/resource/")
ONT = Namespace("http://myolympicgraph.org/ontology/")
WD  = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

# SPARQL HELPERS

def sparql_query(query: str) -> list:
    """Execute une requête SPARQL sur Wikidata, retourne les bindings."""
    try:
        r = requests.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("results", {}).get("bindings", [])
    except Exception as e:
        print(f"    [WARN] SPARQL error: {e}")
        return []


def bindings_to_triples(bindings: list) -> list:
    """Convertit les bindings SPARQL en liste de (s, p, o) strings."""
    triples = []
    for b in bindings:
        s = b.get("s", {}).get("value", "")
        p = b.get("p", {}).get("value", "")
        o = b.get("o", {}).get("value", "")
        o_type = b.get("o", {}).get("type", "uri")
        if s and p and o:
            triples.append((s, p, o, o_type))
    return triples


def is_blacklisted(predicate_uri: str) -> bool:
    return predicate_uri in BLACKLIST_PREDICATES


def is_literal_heavy(predicate_uri: str, value: str) -> bool:
    """Filtre les valeurs littérales trop longues ou peu utiles."""
    if len(value) > 500:
        return True
    # Filtre les IDs externes (contiennent souvent des chiffres longs)
    if value.isdigit() and len(value) > 10:
        return True
    return False


# EXPANSION 1-HOP

def get_aligned_entities(g: Graph) -> list:
    """
    Récupère toutes les entités de ton graphe qui ont un owl:sameAs
    vers Wikidata (résultat du Step 2).
    Retourne une liste de (local_uri, wikidata_uri, label).
    """
    aligned = []
    for s, p, o in g.triples((None, OWL.sameAs, None)):
        if "wikidata.org" in str(o):
            label_triples = list(g.triples((s, RDFS.label, None)))
            label = str(label_triples[0][2]) if label_triples else str(s).split("/")[-1]
            aligned.append((s, o, label))
    return aligned


def expand_1hop(wd_uri: str, limit: int = DEFAULT_LIMIT) -> list:
    """
    1-Hop Expansion : récupère tous les triplets directs d'une entité Wikidata.
    Comme dans la consigne : SELECT ?p ?o WHERE { wd:QXXX ?p ?o }
    """
    query = f"""
SELECT ?p ?o WHERE {{
  <{wd_uri}> ?p ?o .
}}
LIMIT {limit}
"""
    bindings = sparql_query(query)
    triples = []
    for b in bindings:
        p = b.get("p", {}).get("value", "")
        o = b.get("o", {}).get("value", "")
        o_type = b.get("o", {}).get("type", "uri")
        if p and o and not is_blacklisted(p):
            if o_type == "literal" and is_literal_heavy(p, o):
                continue
            triples.append((wd_uri, p, o, o_type))
    return triples


# EXPANSION PAR PRÉDICAT

def expand_by_predicate(pid: str, limit: int = DEFAULT_LIMIT) -> list:
    """
    Predicate-Controlled Expansion :
    Récupère tous les triplets Wikidata utilisant un prédicat donné.
    Comme dans la consigne : SELECT ?s ?p ?o WHERE { ?s wdt:PXXX ?o }
    """
    pred_uri = f"http://www.wikidata.org/prop/direct/{pid}"
    query = f"""
SELECT ?s ?p ?o WHERE {{
  ?s <{pred_uri}> ?o .
  BIND(<{pred_uri}> AS ?p)
}}
LIMIT {limit}
"""
    bindings = sparql_query(query)
    return [
        (
            b.get("s", {}).get("value", ""),
            pred_uri,
            b.get("o", {}).get("value", ""),
            b.get("o", {}).get("type", "uri"),
        )
        for b in bindings
        if b.get("s", {}).get("value") and b.get("o", {}).get("value")
    ]


# EXPANSION 2-HOP

def expand_2hop(intermediate_entities: list, limit: int = 200) -> list:
    """
    2-Hop Expansion : depuis les entités découvertes au 1er hop,
    récupère leurs triplets directs.
    Comme dans la consigne : award → ?p ?o
    """
    all_triples = []
    for wd_uri in intermediate_entities[:MAX_2HOP]:
        query = f"""
SELECT ?p ?o WHERE {{
  <{wd_uri}> ?p ?o .
  FILTER(?p != <http://www.wikidata.org/prop/direct/P18>)
  FILTER(?p != <http://www.wikidata.org/prop/direct/P373>)
}}
LIMIT {limit}
"""
        bindings = sparql_query(query)
        for b in bindings:
            p = b.get("p", {}).get("value", "")
            o = b.get("o", {}).get("value", "")
            o_type = b.get("o", {}).get("type", "uri")
            if p and o and not is_blacklisted(p):
                all_triples.append((wd_uri, p, o, o_type))
        time.sleep(SLEEP_BETWEEN * 0.5)
    return all_triples


# AJOUT AU GRAPHE

def add_triples_to_graph(g: Graph, raw_triples: list) -> int:
    """
    Convertit les raw triples (s, p, o, type) en triplets RDF et les ajoute au graphe.
    Retourne le nombre de triplets ajoutés.
    """
    added = 0
    for s, p, o, o_type in raw_triples:
        try:
            s_uri = URIRef(s)
            p_uri = URIRef(p)
            if o_type == "literal":
                o_node = Literal(o)
            elif o_type in ("uri", "typed-literal"):
                o_node = URIRef(o)
            else:
                o_node = Literal(o)
            g.add((s_uri, p_uri, o_node))
            added += 1
        except Exception:
            continue
    return added


# CLEANING

def clean_graph(g: Graph) -> Graph:
    """
    Nettoyage avant export KGE :
    - Supprime les triplets dupliqués (rdflib gère déjà, mais on vérifie)
    - Supprime les URIs incohérentes
    - Supprime les prédicats blacklistés résiduels
    - Filtre les littéraux trop longs
    """
    print("\n  Cleaning graph...")
    g_clean = Graph()
    g_clean.bind("wd",  WD)
    g_clean.bind("wdt", WDT)
    g_clean.bind("ex",  EX)
    g_clean.bind("ont", ONT)

    removed = 0
    for s, p, o in g:
        # Filtre prédicats blacklistés
        if str(p) in BLACKLIST_PREDICATES:
            removed += 1
            continue
        # Filtre URI malformées
        if str(s).startswith("_:") or str(p).startswith("_:"):
            removed += 1
            continue
        # Filtre littéraux trop longs
        if isinstance(o, Literal) and len(str(o)) > 1000:
            removed += 1
            continue
        g_clean.add((s, p, o))

    print(f"  Removed {removed} unwanted triples during cleaning.")
    return g_clean


# STATS

def print_stats(g: Graph, label: str = "Graph"):
    """Affiche les statistiques du graphe."""
    subjects   = set(s for s, _, _ in g)
    predicates = set(p for _, p, _ in g)
    objects    = set(o for _, _, o in g if isinstance(o, URIRef))
    entities   = subjects | objects

    print(f"\n  [{label}]")
    print(f"    Triplets (50k-200k) : {len(g):>10,}")
    print(f"    Entites   (5k-30k)  : {len(entities):>10,}")
    print(f"    Relations (50-200)  : {len(predicates):>10,}")


# PIPELINE PRINCIPAL

def expand(input_ttl: str, output_nt: str, limit: int = DEFAULT_LIMIT):

    print("\n" + "="*62)
    print("  STEP 4 - KB Expansion via SPARQL (Wikidata)")
    print("="*62)

    # Charge graphe Step 3
    g = Graph()
    g.parse(input_ttl, format="turtle")
    g.bind("wd",  WD)
    g.bind("wdt", WDT)
    g.bind("ex",  EX)
    g.bind("ont", ONT)
    print(f"\n  Graphe initial charge : {len(g)} triplets")

    # Entités alignées avec Wikidata
    aligned = get_aligned_entities(g)
    print(f"  Entites alignees avec Wikidata : {len(aligned)}")

    if not aligned:
        print("\n  ATTENTION : aucune entite avec owl:sameAs trouve.")
        print("  Assure-toi d'avoir lance align.py (Step 2) avant.")
        return

    # Limite le nombre d'entités pour l'expansion
    aligned_to_expand = aligned[:MAX_ENTITIES]
    print(f"  Entites a expander (1-hop) : {len(aligned_to_expand)}")

    # ──────────────────────────────────────
    # PHASE 1 — 1-Hop Expansion
    # ──────────────────────────────────────
    print(f"\n  ── Phase 1 : 1-Hop Expansion ──")
    intermediate_uris = set()
    total_added_phase1 = 0

    for i, (local_uri, wd_uri, label) in enumerate(aligned_to_expand):
        print(f"  [{i+1}/{len(aligned_to_expand)}] '{label}' -> {str(wd_uri).split('/')[-1]} ...", end=" ", flush=True)
        raw = expand_1hop(str(wd_uri), limit=limit)
        added = add_triples_to_graph(g, raw)
        total_added_phase1 += added
        print(f"{added} triplets")

        # Collecte les entités Wikidata pour le 2-hop
        for _, _, o, o_type in raw:
            if o_type == "uri" and "wikidata.org/entity/Q" in o:
                intermediate_uris.add(o)

        time.sleep(SLEEP_BETWEEN)

    print(f"\n  Phase 1 terminee : +{total_added_phase1:,} triplets")
    print(f"  Entites intermediaires collectees : {len(intermediate_uris):,}")

    # ──────────────────────────────────────
    # PHASE 2 — Predicate-Controlled Expansion
    # ──────────────────────────────────────
    print(f"\n  ── Phase 2 : Predicate-Controlled Expansion ──")
    total_added_phase2 = 0

    for pid, pred_label in PRIORITY_PREDICATES:
        if len(g) >= TARGET_MAX:
            print(f"  Objectif max atteint, arret phase 2.")
            break
        print(f"  Expanding wdt:P{pid} ({pred_label}) ...", end=" ", flush=True)
        raw = expand_by_predicate(pid, limit=limit)
        added = add_triples_to_graph(g, raw)
        total_added_phase2 += added
        print(f"{added} triplets (total: {len(g):,})")
        time.sleep(SLEEP_BETWEEN)

    print(f"\n  Phase 2 terminee : +{total_added_phase2:,} triplets")

    # ──────────────────────────────────────
    # PHASE 3 — 2-Hop Expansion
    # ──────────────────────────────────────
    if len(g) < TARGET_MIN:
        print(f"\n  ── Phase 3 : 2-Hop Expansion ──")
        print(f"  ({len(intermediate_uris):,} entites intermediaires disponibles)")
        inter_list = list(intermediate_uris)
        raw = expand_2hop(inter_list, limit=200)
        added = add_triples_to_graph(g, raw)
        print(f"  Phase 3 terminee : +{added:,} triplets")
    else:
        print(f"\n  Phase 3 : skippee (objectif min deja atteint avec {len(g):,} triplets)")

    # ──────────────────────────────────────
    # CLEANING
    # ──────────────────────────────────────
    g = clean_graph(g)


    print_stats(g, label="Graphe final apres expansion")

    # ──────────────────────────────────────
    # SAUVEGARDE
    # ──────────────────────────────────────
    import os
    os.makedirs("././kg_artifacts", exist_ok=True)
    g.serialize(output_nt, format="nt")
    print(f"\n  Graphe sauvegarde -> {output_nt}")

    # Sauvegarde aussi en TTL pour lisibilite
    ttl_path = output_nt.replace(".nt", ".ttl")
    g.serialize(ttl_path, format="turtle")
    print(f"  Copie TTL          -> {ttl_path}")

    print(f"\n{'='*62}")
    print(f"  Step 4 termine")
    print(f"  Total triplets : {len(g):,}")
    print(f"{'='*62}\n")


# CLI

if __name__ == "__main__":
    import os
    os.makedirs("././kg_artifacts", exist_ok=True)

    parser = argparse.ArgumentParser(description="Step 4: KB Expansion via SPARQL")
    parser.add_argument("--graph",  default="././kg_artifacts/predicate_alignment.ttl",
                        help="Input TTL from Step 3")
    parser.add_argument("--out",    default="././kg_artifacts/expanded.nt",
                        help="Output NT file (expanded KB)")
    parser.add_argument("--limit",  type=int, default=DEFAULT_LIMIT,
                        help="Max triples per SPARQL query (default: 500)")
    args = parser.parse_args()

    expand(
        input_ttl  = args.graph,
        output_nt  = args.out,
        limit      = args.limit,
    )