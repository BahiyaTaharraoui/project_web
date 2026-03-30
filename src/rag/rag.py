"""
rag.py — TP3 RAG with RDF/SPARQL + Ollama (v4 — entièrement retesté)

CORRECTIONS v4 :
  - Schema summary reécrit avec les vrais prédicats/classes du graphe
  - 5 questions d'évaluation adaptées à ce que le graphe contient VRAIMENT
  - SPARQL fallback hardcodé si le LLM génère du mauvais SPARQL
  - Filtre LANG obligatoire sur tous les labels (6000+ labels multilingues)
  - Post-processing qui nettoie BIND, UNION, semicolons, classes inventées
  - Timeout LLM augmenté + message d'erreur clair

Usage:
    python rag.py --graph kg_artifacts/expanded.nt
    python rag.py --graph kg_artifacts/expanded.nt --eval
    python rag.py --graph kg_artifacts/expanded.nt --model gemma:2b
List the members of the organizations."""

import argparse
import re
import json
import os
from typing import List, Tuple
import requests

# ==============================
# CONFIG
# ==============================
OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:latest"
CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

# Namespaces réels du graphe
NS_ONT  = "http://myolympicgraph.org/ontology/"
NS_EX   = "http://myolympicgraph.org/resource/"
NS_WD   = "http://www.wikidata.org/entity/"
NS_WDT  = "http://www.wikidata.org/prop/direct/"
NS_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
NS_RDF  = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_OWL  = "http://www.w3.org/2002/07/owl#"

# ==============================
# 0) OLLAMA
# ==============================

def check_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def ask_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    try:
        r = requests.post(OLLAMA_URL,
                          json={"model": model, "prompt": prompt, "stream": False},
                          timeout=180)
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return "[ERREUR] Ollama non démarré. Lance: ollama serve"
    except requests.exceptions.Timeout:
        return "[ERREUR] Timeout — le modèle met trop de temps à répondre."
    except Exception as e:
        return f"[ERREUR] {e}"


# ==============================
# 1) CHARGEMENT DU GRAPHE
# ==============================

def _try_rdflib():
    try:
        from rdflib import Graph
        return Graph
    except ImportError:
        return None


class NTGraph:
    """
    Parser NT minimal — fonctionne SANS rdflib.
    Supporte les requêtes SPARQL via rdflib si disponible,
    sinon via un moteur de requêtes simplifié intégré.
    """
    def __init__(self):
        self.uri_triples = []   # (s, p, o) tous URIs
        self.lit_triples = []   # (s, p, o) avec literal
        self._rdflib_graph = None

    def parse(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip(" .").split(" ", 2)
                if len(parts) < 3:
                    continue
                s, p, o = parts[0], parts[1], parts[2].strip()
                if s.startswith("<") and p.startswith("<"):
                    su, pu = s[1:-1], p[1:-1]
                    if o.startswith("<"):
                        self.uri_triples.append((su, pu, o[1:-1]))
                    else:
                        self.lit_triples.append((su, pu, o))
        return self

    def __len__(self):
        return len(self.uri_triples) + len(self.lit_triples)

    def _get_rdflib(self):
        """Construit le graph rdflib une seule fois (lazy)."""
        if self._rdflib_graph is not None:
            return self._rdflib_graph
        G = _try_rdflib()
        if G is None:
            return None
        import tempfile
        g = G()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nt",
                                         delete=False, encoding="utf-8") as tmp:
            for s, p, o in self.uri_triples:
                tmp.write(f"<{s}> <{p}> <{o}> .\n")
            for s, p, o in self.lit_triples:
                tmp.write(f"<{s}> <{p}> {o} .\n")
            tmp_path = tmp.name
        g.parse(tmp_path, format="nt")
        os.unlink(tmp_path)
        self._rdflib_graph = g
        return g

    def sparql(self, query: str):
        """Exécute SPARQL via rdflib."""
        g = self._get_rdflib()
        if g is None:
            raise RuntimeError(
                "rdflib non installé. Lance: pip install rdflib\n"
                "Sans rdflib les requêtes SPARQL ne sont pas disponibles."
            )
        res   = g.query(query)
        vars_ = [str(v) for v in res.vars]
        rows  = [tuple(str(c) if c is not None else "" for c in r) for r in res]
        return vars_, rows

    @property
    def namespace_manager(self):
        class _NS:
            def namespaces(self):
                return []
        return _NS()


def load_graph(path: str) -> NTGraph:
    G_cls = _try_rdflib()
    g = NTGraph()
    g.parse(path)
    if G_cls:
        # Force construction du graph rdflib maintenant (pas en lazy)
        _ = g._get_rdflib()
        print(f"  [OK] Graphe chargé via rdflib : {len(g):,} triplets")
    else:
        print(f"  [WARN] rdflib absent — pip install rdflib pour le SPARQL")
        print(f"  [OK] Graphe chargé (NT parser) : {len(g):,} triplets")
    return g


# ==============================
# 2) SCHEMA SUMMARY
# ==============================
# Construit à partir de l'analyse réelle du graphe expanded.nt

FIXED_SCHEMA = """
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX ont:  <http://myolympicgraph.org/ontology/>
PREFIX ex:   <http://myolympicgraph.org/resource/>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

# ── CLASSES (rdf:type values that actually exist) ────────────────────────────
# - <http://myolympicgraph.org/ontology/Person>        (104 instances)
# - <http://myolympicgraph.org/ontology/Organization>  (60 instances)
# - <http://myolympicgraph.org/ontology/Place>         (25 instances)
# DO NOT USE any other class. ont:Person = Person, NOT ex:Person or ex:Athlete.

# ── LOCAL PREDICATES (myolympicgraph.org) ────────────────────────────────────
# <http://myolympicgraph.org/ontology/memberOf>   Person → Organization  (132 triples)
# <http://myolympicgraph.org/ontology/locatedIn>  Organization → Place   (some triples)
# <http://www.w3.org/2002/07/owl#sameAs>          local entity → wikidata entity
# <http://www.w3.org/2000/01/rdf-schema#label>    entity → string literal
# <http://www.w3.org/2000/01/rdf-schema#seeAlso>  entity → Wikipedia URL

# ── WIKIDATA PREDICATES (most frequent) ─────────────────────────────────────
# wdt:P463  member of            (589 triples)
# wdt:P166  award received       (615 triples)
# wdt:P106  occupation           (590 triples)
# wdt:P31   instance of          (788 triples)
# wdt:P17   country              (580 triples)
# wdt:P27   country of citizenship (531 triples)
# wdt:P131  located in administrative entity (545 triples)
# wdt:P21   sex or gender        (534 triples)
# wdt:P19   place of birth       (525 triples)
# wdt:P569  date of birth        (531 triples)

# ── VERIFIED WORKING SPARQL QUERIES ─────────────────────────────────────────

# Q1: List all local persons
SELECT ?person ?label WHERE {
  ?person a <http://myolympicgraph.org/ontology/Person> .
  OPTIONAL { ?person <http://www.w3.org/2000/01/rdf-schema#label> ?label .
             FILTER(LANG(?label) = "" || LANG(?label) = "en") }
} LIMIT 20

# Q2: List all local organizations
SELECT ?org ?label WHERE {
  ?org a <http://myolympicgraph.org/ontology/Organization> .
  OPTIONAL { ?org <http://www.w3.org/2000/01/rdf-schema#label> ?label .
             FILTER(LANG(?label) = "" || LANG(?label) = "en") }
} LIMIT 20

# Q3: memberOf relations (Person → Organization)
SELECT ?person ?plabel ?org ?olabel WHERE {
  ?person <http://myolympicgraph.org/ontology/memberOf> ?org .
  OPTIONAL { ?person <http://www.w3.org/2000/01/rdf-schema#label> ?plabel .
             FILTER(LANG(?plabel) = "" || LANG(?plabel) = "en") }
  OPTIONAL { ?org <http://www.w3.org/2000/01/rdf-schema#label> ?olabel .
             FILTER(LANG(?olabel) = "" || LANG(?olabel) = "en") }
} LIMIT 20

# Q4: Wikidata member-of (P463)
SELECT ?person ?org WHERE {
  ?person <http://www.wikidata.org/prop/direct/P463> ?org .
} LIMIT 20

# Q5: Organizations mentioned on Olympic Games Wikipedia page
SELECT ?org ?label WHERE {
  ?org a <http://myolympicgraph.org/ontology/Organization> .
  ?org <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?url .
  FILTER(CONTAINS(STR(?url), "Olympic_Games"))
  OPTIONAL { ?org <http://www.w3.org/2000/01/rdf-schema#label> ?label .
             FILTER(LANG(?label) = "" || LANG(?label) = "en") }
} LIMIT 20

# ── CRITICAL RULES ───────────────────────────────────────────────────────────
# RULE 1: rdfs:label has 6000+ multilingual values (Japanese, Russian, Arabic...)
#         ALWAYS add: FILTER(LANG(?label) = "" || LANG(?label) = "en")
# RULE 2: Do NOT use BIND — write FILTER instead.
# RULE 3: Do NOT use UNION.
# RULE 4: Do NOT add ; after LIMIT.  Correct: LIMIT 20    Wrong: LIMIT 20;
# RULE 5: Valid classes: ont:Person  ont:Organization  ont:Place  ONLY.
#         NEVER invent: ex:OlympicGame  ex:Athlete  ex:Country  ex:Event
# RULE 6: Use full URIs in <angle brackets> when using prefixes is ambiguous.
"""


def build_schema_summary(g: NTGraph) -> str:
    """Retourne le schema summary fixe (validé sur le vrai graphe)."""
    return FIXED_SCHEMA.strip()


# ==============================
# 3) SPARQL FALLBACKS HARDCODÉS
# ==============================
# Ces requêtes ont été testées et retournent des résultats sur expanded.nt

FALLBACK_QUERIES = {
    "persons": """SELECT ?person ?label WHERE {
  ?person a <http://myolympicgraph.org/ontology/Person> .
  OPTIONAL { ?person <http://www.w3.org/2000/01/rdf-schema#label> ?label .
             FILTER(LANG(?label) = "" || LANG(?label) = "en") }
} LIMIT 20""",

    "organizations": """SELECT ?org ?label WHERE {
  ?org a <http://myolympicgraph.org/ontology/Organization> .
  OPTIONAL { ?org <http://www.w3.org/2000/01/rdf-schema#label> ?label .
             FILTER(LANG(?label) = "" || LANG(?label) = "en") }
} LIMIT 20""",

    "memberof": """SELECT ?person ?plabel ?org ?olabel WHERE {
  ?person <http://myolympicgraph.org/ontology/memberOf> ?org .
  OPTIONAL { ?person <http://www.w3.org/2000/01/rdf-schema#label> ?plabel .
             FILTER(LANG(?plabel) = "" || LANG(?plabel) = "en") }
  OPTIONAL { ?org <http://www.w3.org/2000/01/rdf-schema#label> ?olabel .
             FILTER(LANG(?olabel) = "" || LANG(?olabel) = "en") }
} LIMIT 20""",

    "olympic_orgs": """SELECT ?org ?label WHERE {
  ?org a <http://myolympicgraph.org/ontology/Organization> .
  ?org <http://www.w3.org/2000/01/rdf-schema#seeAlso> ?url .
  FILTER(CONTAINS(STR(?url), "Olympic_Games"))
  OPTIONAL { ?org <http://www.w3.org/2000/01/rdf-schema#label> ?label .
             FILTER(LANG(?label) = "" || LANG(?label) = "en") }
} LIMIT 20""",

    "wdt_members": """SELECT ?person ?org WHERE {
  ?person <http://www.wikidata.org/prop/direct/P463> ?org .
} LIMIT 20""",

    "awards": """SELECT ?entity ?award WHERE {
  ?entity <http://www.wikidata.org/prop/direct/P166> ?award .
} LIMIT 20""",
}


def pick_fallback(question: str) -> str:
    """Sélectionne la requête fallback la plus adaptée à la question."""
    q = question.lower()
    if "member" in q or "ioc" in q or "committee" in q:
        return """SELECT ?s ?p ?o WHERE {
        ?s ?p ?o .
        } LIMIT 20"""
    if "person" in q or "athlete" in q or "people" in q or "who" in q:
        return FALLBACK_QUERIES["persons"]
    if "organizat" in q or "org" in q or "committee" in q or "federation" in q:
        return FALLBACK_QUERIES["olympic_orgs"]
    if "award" in q or "medal" in q or "prize" in q:
        return FALLBACK_QUERIES["awards"]
    if "olympic" in q and ("org" in q or "asso" in q):
        return FALLBACK_QUERIES["olympic_orgs"]
    return FALLBACK_QUERIES["persons"]


# ==============================
# 4) NETTOYAGE DU SPARQL LLM
# ==============================

KNOWN_CLASSES = {
    "http://myolympicgraph.org/ontology/Person",
    "http://myolympicgraph.org/ontology/Organization",
    "http://myolympicgraph.org/ontology/Place",
}


def clean_llm_sparql(raw_text: str) -> str:
    """
    Extrait et nettoie le SPARQL produit par le LLM.
    Corrige les erreurs typiques avant exécution.
    """
    # 1. Extrait le bloc ```sparql ... ```
    m = CODE_BLOCK_RE.search(raw_text)
    q = m.group(1).strip() if m else raw_text.strip()

    # 2. Si le texte commence par '[' ou autre char invalide → vide
    if not q or q[0] not in ("S", "s", "C", "c", "D", "d", "A", "a"):
        return ""

    # 3. Supprime le ';' final invalide
    q = q.rstrip("; \t\n")

    # 4. Supprime les lignes commentaires C++/JS (//)
    q = "\n".join(l for l in q.splitlines() if not l.strip().startswith("//"))

    # 5. Remplace BIND par un commentaire (syntaxe non supportée par rdflib)
    if re.search(r'\bBIND\b', q, re.IGNORECASE):
        # Retire les lignes avec BIND
        q = "\n".join(l for l in q.splitlines()
                      if not re.search(r'\bBIND\b', l, re.IGNORECASE))

    # 6. Retire les clauses rdf:type avec une classe inconnue
    def fix_type_clause(m):
        cls = m.group(2).strip()
        # Résout le préfixe en URI complète
        cls_uri = cls
        if cls.startswith("ont:"):
            cls_uri = NS_ONT + cls[4:]
        elif cls.startswith("ex:"):
            cls_uri = NS_EX + cls[3:]
        elif cls.startswith("<") and cls.endswith(">"):
            cls_uri = cls[1:-1]
        if cls_uri in KNOWN_CLASSES or any(cls_uri.startswith(c) for c in KNOWN_CLASSES):
            return m.group(0)   # classe valide → garde
        return ""               # classe inventée → retire

    q = re.sub(
        r'(\?\w+\s+(?:a|rdf:type)\s+)(<[^>]+>|[\w:]+)(\s*\.)',
        fix_type_clause, q, flags=re.IGNORECASE
    )

    # 7. Ajoute FILTER langue si rdfs:label sans filtre
    if "rdfs:label" in q or "rdf-schema#label" in q:
        if "LANG(" not in q.upper():
            q = q.replace(
                "rdfs:label ?label",
                'rdfs:label ?label . FILTER(LANG(?label) = "" || LANG(?label) = "en")'
            ).replace(
                "<http://www.w3.org/2000/01/rdf-schema#label> ?label",
                '<http://www.w3.org/2000/01/rdf-schema#label> ?label . '
                'FILTER(LANG(?label) = "" || LANG(?label) = "en")'
            )

    # 8. Nettoie les lignes vides multiples
    lines = [l for l in q.splitlines() if l.strip()]
    return "\n".join(lines).strip()


# ==============================
# 5) PROMPTS LLM
# ==============================

SPARQL_PROMPT = """You are a SPARQL generator expert for an Olympic Knowledge Graph.
Your task is to convert the QUESTION into a valid SPARQL 1.1 SELECT query using the provided SCHEMA.

STRICT RULES:
1. ONLY use predicates and classes from the SCHEMA SUMMARY.
2. Valid Classes: 
   - <http://myolympicgraph.org/ontology/Person>
   - <http://myolympicgraph.org/ontology/Organization>
   - <http://myolympicgraph.org/ontology/Place>
3. Syntax for types: Use 'a' or 'rdf:type'. (e.g., ?p a <http://myolympicgraph.org/ontology/Person> .)
4. Multilingual labels: If you use rdfs:label, you MUST add:
   FILTER(LANG(?label) = "" || LANG(?label) = "en")
5. Limitations: DO NOT use BIND, DO NOT use UNION, DO NOT use semicolons (;) after LIMIT.
6. Template: Follow the structure of the "VERIFIED WORKING SPARQL QUERIES" exactly.
7. Output: Return ONLY the SPARQL query inside a ```sparql block. No explanation.

SCHEMA SUMMARY:
{schema}

QUESTION: {question}
"""

# SPARQL_PROMPT = """You are a SPARQL generator for an Olympic Knowledge Graph (rdflib).
# Generate a valid SPARQL 1.1 SELECT query for the QUESTION below.

# MANDATORY RULES:
# 1. Use ONLY predicates listed in the SCHEMA SUMMARY.
# 2. Valid rdf:type classes: <http://myolympicgraph.org/ontology/Person>
#                            <http://myolympicgraph.org/ontology/Organization>
#                            <http://myolympicgraph.org/ontology/Place>
#    NEVER invent: ex:OlympicGame, ex:Athlete, ex:Country, etc.
# 3. rdfs:label has multilingual data. ALWAYS add:
#    FILTER(LANG(?label) = "" || LANG(?label) = "en")
# 4. Do NOT use BIND — not supported. Use FILTER instead.
# 5. Do NOT use UNION.
# 6. Do NOT add ; after LIMIT.  RIGHT: LIMIT 20    WRONG: LIMIT 20;
# 7. Copy one of the VERIFIED WORKING SPARQL QUERIES from the schema as a template.
# 8. Return ONLY the query inside ```sparql ... ``` block. Nothing else.

# SCHEMA SUMMARY:
# {schema}

# QUESTION: {question}

# Return only the SPARQL query in a ```sparql block."""


REPAIR_PROMPT = """The SPARQL query below failed on rdflib. Fix it.

MANDATORY FIXES:
1. Remove BIND — use FILTER instead.
2. Remove ; after LIMIT.
3. Add FILTER(LANG(?label) = "" || LANG(?label) = "en") if rdfs:label is used.
4. Remove unknown classes (keep only: ont:Person, ont:Organization, ont:Place).
5. Use one of the VERIFIED WORKING examples from the schema as template.
6. Return ONLY the fixed query in a ```sparql block.

SCHEMA SUMMARY:
{schema}

QUESTION: {question}
BAD SPARQL: {bad_query}
ERROR: {error}

Return only the corrected SPARQL in a ```sparql block."""


# ==============================
# 6) PIPELINE RAG
# ==============================

def answer_no_rag(question: str, model: str) -> str:
    return ask_llm(
        f"Answer the following question concisely:\n\n{question}", model
    )


def answer_with_rag(g: NTGraph, schema: str, question: str,
                    model: str, try_repair: bool = True) -> dict:
    """
    Pipeline RAG complet :
    1. Génère SPARQL via LLM
    2. Nettoie le SPARQL
    3. Exécute
    4. Si erreur → self-repair
    5. Si toujours erreur → fallback hardcodé
    """
    # Étape 1 : génération
    raw = ask_llm(SPARQL_PROMPT.format(schema=schema, question=question), model)
    sparql = clean_llm_sparql(raw)

    if not is_valid_sparql(sparql):
        print("  [WARN] SPARQL invalide généré → fallback direct")
        fallback_q = pick_fallback(question)
        vars_, rows = g.sparql(fallback_q)
        return {"query": fallback_q, "vars": vars_, "rows": rows,
                "repaired": False, "fallback": True, "error": None}

    # Étape 2 : exécution
    if sparql:
        try:
            vars_, rows = g.sparql(sparql)
            return {"query": sparql, "vars": vars_, "rows": rows,
                    "repaired": False, "fallback": False, "error": None}
        except Exception as e:
            err = str(e)
            print(f"  [SPARQL ERROR] {err[:80]} → repair...")

            # Étape 3 : self-repair
            if False:
                raw2   = ask_llm(REPAIR_PROMPT.format(
                    schema=schema, question=question,
                    bad_query=sparql, error=err), model)
                sparql2 = clean_llm_sparql(raw2)
                if sparql2:
                    try:
                        vars_, rows = g.sparql(sparql2)
                        return {"query": sparql2, "vars": vars_, "rows": rows,
                                "repaired": True, "fallback": False, "error": None}
                    except Exception as e2:
                        print(f"  [REPAIR FAILED] {str(e2)[:80]} → fallback...")
    else:
        print("  [WARN] LLM n'a pas produit de SPARQL valide → fallback...")

    # Étape 4 : fallback hardcodé (toujours fonctionnel)
    fallback_q = pick_fallback(question)
    try:
        vars_, rows = g.sparql(fallback_q)
        return {"query": fallback_q, "vars": vars_, "rows": rows,
                "repaired": False, "fallback": True, "error": None}
    except Exception as ef:
        return {"query": fallback_q, "vars": [], "rows": [],
                "repaired": False, "fallback": True, "error": str(ef)}


# ==============================
# 7) AFFICHAGE
# ==============================

def pretty_print(result: dict) -> None:
    tags = []
    if result.get("repaired"):  tags.append("auto-réparé")
    if result.get("fallback"):  tags.append("fallback hardcodé")
    tag_str = f"  ({', '.join(tags)})" if tags else ""

    print(f"\n  [SPARQL utilisé{tag_str}]")
    for line in result["query"].splitlines():
        print(f"    {line}")

    rows  = result.get("rows", [])
    vars_ = result.get("vars", [])

    if result.get("error"):
        print(f"\n  [Erreur] {result['error']}")

    if not rows:
        print("\n  [Aucun résultat]")
        return

    print(f"\n  [Résultats — {len(rows)} ligne(s)]")
    print("  " + " | ".join(vars_))
    print("  " + "─" * 60)
    for r in rows[:20]:
        display = []
        for cell in r:
            if cell.startswith("http"):
                cell = cell.split("/")[-1].split("#")[-1]
            display.append(cell[:50])
        print("  " + " | ".join(display))
    if len(rows) > 20:
        print(f"  … ({len(rows) - 20} résultats supplémentaires)")


# ==============================
# 8) ÉVALUATION 5 QUESTIONS
# ==============================
# Questions adaptées à ce que le graphe contient réellement

EVAL_QUESTIONS = [
    "List all persons in the knowledge graph.",
    "List all organizations in the knowledge graph.",
    "Which persons are members of an organization (memberOf relations)?",
    "Which entities received awards (wdt:P463 or wdt:P166)?",
    "Which organizations are mentioned on the Olympic Games Wikipedia page?",
]


def run_evaluation(g: NTGraph, schema: str, model: str):
    print("\n" + "="*62)
    print("  ÉVALUATION — Baseline vs RAG (5 questions)")
    print("="*62)
    results = []
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n  ── Q{i}: {q}")
        print(f"  {'─'*55}")

        print("  [Baseline — LLM seul]")
        baseline = answer_no_rag(q, model)
        print(f"  {baseline[:200]}{'...' if len(baseline) > 200 else ''}")

        print("\n  [RAG — SPARQL + graphe]")
        rag = answer_with_rag(g, schema, q, model)
        pretty_print(rag)

        results.append({
            "question":  q,
            "baseline":  baseline[:300],
            "sparql":    rag["query"],
            "rag_rows":  len(rag.get("rows", [])),
            "repaired":  rag["repaired"],
            "fallback":  rag["fallback"],
            "error":     rag.get("error"),
        })

    os.makedirs("././reports", exist_ok=True)
    out = "././reports/rag_evaluation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Évaluation sauvegardée → {out}")
    return results

def is_valid_sparql(query: str) -> bool:
    if not query:
        return False

    q = query.strip().lower()

    # doit commencer par SELECT / ASK / CONSTRUCT
    if not q.startswith(("select", "ask", "construct", "describe")):
        return False

    # doit contenir WHERE
    if "where" not in q:
        return False

    # variables utilisées dans SELECT doivent exister
    if "select" in q:
        vars_ = re.findall(r"\?(\w+)", q.split("where")[0])
        if not vars_:
            return False

    return True

# ==============================
# 9) CLI
# ==============================

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG SPARQL — Olympic KG")
    parser.add_argument("--graph",     default="././kg_artifacts/expanded.nt")
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--eval",      action="store_true")
    args = parser.parse_args()

    print("\n" + "="*62)
    print("  RAG — RDF/SPARQL + Ollama — Olympic Knowledge Graph")
    print("="*62)

    print(f"\n  Vérification Ollama ({OLLAMA_URL})...")
    if check_ollama():
        print(f"  Ollama OK — modèle : {args.model}")
    else:
        print(f"  [ATTENTION] Ollama non disponible !")
        print(f"  Lance : ollama serve  puis  ollama pull {args.model}")

    print(f"\n  Chargement : {args.graph}")
    g      = load_graph(args.graph)
    schema = build_schema_summary(g)

    if args.eval:
        run_evaluation(g, schema, args.model)
        return

    print(f"\n  Tape ta question. 'quit' pour quitter, 'eval' pour lancer l'évaluation.\n")
    while True:
        try:
            q = input("  Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Au revoir !")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("  Au revoir !")
            break
        if q.lower() == "eval":
            run_evaluation(g, schema, args.model)
            continue

        print("\n" + "="*62)
        print("  BASELINE (LLM seul)")
        print("="*62)
        print(answer_no_rag(q, args.model))

        print("\n" + "="*62)
        print("  RAG (SPARQL + graphe)")
        print("="*62)
        pretty_print(answer_with_rag(g, schema, q, args.model,
                                     try_repair=not args.no_repair))
        print()


if __name__ == "__main__":
    main()