"""
build_graph.py — TP1 Step 1 (complet)
Construit le graphe RDF initial depuis extracted_knowledge.csv.

Ce script fait :
  - Définition des classes ontologiques (PERSON, ORG, GPE)
  - Définition des propriétés avec domain/range
  - Ajout des entités avec labels, types, source URL
  - Inférence de relations entre entités co-présentes dans la même page source :
      PERSON  --[memberOf]-->   ORG
      PERSON  --[basedIn]-->    GPE
      ORG     --[locatedIn]-->  GPE
  - Dédoublonnage des entités similaires (ex: "IOC" et "International Olympic Committee")
  - Sérialisation en Turtle (.ttl)

Usage:
    python build_graph.py
"""

import re
import pandas as pd
from collections import defaultdict
from rdflib import Graph, Literal, RDF, RDFS, OWL, Namespace, URIRef, XSD

# ==============================
# CONFIG
# ==============================
CSV_PATH       = "././data/extracted_knowledge.csv"
OUTPUT_TTL     = "././kg_artifacts/graph.ttl"

EX  = Namespace("http://myolympicgraph.org/resource/")
ONT = Namespace("http://myolympicgraph.org/ontology/")

# Nombre max de relations inférées par page (évite l'explosion combinatoire)
MAX_RELATIONS_PER_PAGE = 30

# ==============================
# HELPERS
# ==============================

def uriize(label: str) -> str:
    """Transforme un label texte en fragment d'URI valide."""
    s = label.strip().replace(" ", "_")
    s = re.sub(r'[^\w]', '', s)
    return s or "Unknown"


def normalize_label(label: str) -> str:
    """Normalise un label pour comparaison (lowercase, sans articles)."""
    label = label.lower().strip()
    for prefix in ["the ", "a ", "an ", "les ", "le ", "la "]:
        if label.startswith(prefix):
            label = label[len(prefix):]
    return label


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les entités quasi-dupliquées :
    - même type
    - l'une est une sous-chaîne normalisée de l'autre
    Garde la version la plus longue (plus précise).
    """
    df = df.copy()
    df['norm'] = df['entity'].apply(normalize_label)
    # Groupe par type, élimine les norms qui sont substring d'une autre
    to_drop = set()
    for t in df['type'].unique():
        sub = df[df['type'] == t].copy()
        norms = sub['norm'].tolist()
        for i, n1 in enumerate(norms):
            for j, n2 in enumerate(norms):
                if i != j and n1 in n2 and len(n1) < len(n2):
                    # n1 est sous-chaîne de n2 → on garde n2, on drop n1
                    to_drop.add(sub.index[i])
    df = df.drop(index=to_drop)
    df = df.drop(columns=['norm'])
    # Déduplique aussi les labels exacts
    df = df.drop_duplicates(subset=['entity', 'type'])
    return df.reset_index(drop=True)


def build_ontology(g: Graph):
    """Définit les classes et propriétés de l'ontologie."""

    # --- Classes ---
    classes = {
        ONT.Person: "A human being mentioned in the Olympic context",
        ONT.Organization: "A committee, federation, club or institution",
        ONT.Place: "A geopolitical entity: city, country or region",
    }
    for cls_uri, comment in classes.items():
        g.add((cls_uri, RDF.type,       RDFS.Class))
        g.add((cls_uri, RDF.type,       OWL.Class))
        g.add((cls_uri, RDFS.comment,   Literal(comment, lang="en")))
        g.add((cls_uri, RDFS.subClassOf, OWL.Thing))

    # --- Propriétés ---
    # memberOf : Person → Organization
    g.add((ONT.memberOf,    RDF.type,       OWL.ObjectProperty))
    g.add((ONT.memberOf,    RDFS.label,     Literal("member of", lang="en")))
    g.add((ONT.memberOf,    RDFS.comment,   Literal("A person is affiliated with an organization", lang="en")))
    g.add((ONT.memberOf,    RDFS.domain,    ONT.Person))
    g.add((ONT.memberOf,    RDFS.range,     ONT.Organization))

    # basedIn : Person → Place
    g.add((ONT.basedIn,     RDF.type,       OWL.ObjectProperty))
    g.add((ONT.basedIn,     RDFS.label,     Literal("based in", lang="en")))
    g.add((ONT.basedIn,     RDFS.comment,   Literal("A person is associated with a geopolitical location", lang="en")))
    g.add((ONT.basedIn,     RDFS.domain,    ONT.Person))
    g.add((ONT.basedIn,     RDFS.range,     ONT.Place))

    # locatedIn : Organization → Place
    g.add((ONT.locatedIn,   RDF.type,       OWL.ObjectProperty))
    g.add((ONT.locatedIn,   RDFS.label,     Literal("located in", lang="en")))
    g.add((ONT.locatedIn,   RDFS.comment,   Literal("An organization is located in a place", lang="en")))
    g.add((ONT.locatedIn,   RDFS.domain,    ONT.Organization))
    g.add((ONT.locatedIn,   RDFS.range,     ONT.Place))

    # mentionedIn : Any → URL source (datatype property)
    g.add((ONT.mentionedIn, RDF.type,       OWL.DatatypeProperty))
    g.add((ONT.mentionedIn, RDFS.label,     Literal("mentioned in", lang="en")))
    g.add((ONT.mentionedIn, RDFS.comment,   Literal("Source Wikipedia page where the entity was extracted", lang="en")))

    print("  Ontology: 3 classes + 4 properties defined.")

# ==============================
# STEP 1B — Entités
# ==============================

# Mapping NER type → classe ontologique
TYPE_MAP = {
    "PERSON": ONT.Person,
    "ORG":    ONT.Organization,
    "GPE":    ONT.Place,
}

def add_entities(g: Graph, df: pd.DataFrame) -> dict:
    """
    Ajoute chaque entité au graphe.
    Retourne un dict {entity_label: entity_uri} pour la suite.
    """
    entity_index = {}

    for _, row in df.iterrows():
        label = str(row['entity']).strip()
        ner_t = str(row['type']).strip()
        url   = str(row['url']).strip()

        uri_frag   = uriize(label)
        entity_uri = EX[uri_frag]
        ont_class  = TYPE_MAP.get(ner_t, ONT.Person)

        g.add((entity_uri, RDF.type,        ont_class))
        g.add((entity_uri, RDFS.label,      Literal(label, datatype=XSD.string)))
        g.add((entity_uri, ONT.mentionedIn, Literal(url,   datatype=XSD.anyURI)))
        g.add((entity_uri, RDFS.seeAlso,    URIRef(url)))

        entity_index[label] = entity_uri

    print(f"  Entities: {len(entity_index)} unique entities added.")
    return entity_index


# ==============================
# STEP 1C — Relations inférées
# ==============================

def infer_relations(g: Graph, df: pd.DataFrame, entity_index: dict):
    """
    Pour chaque page source, crée des relations entre entités co-présentes :
      PERSON  --memberOf-->  ORG
      PERSON  --basedIn-->   GPE
      ORG     --locatedIn--> GPE
    On limite à MAX_RELATIONS_PER_PAGE pour rester raisonnable.
    """
    relation_count = 0

    for url, group in df.groupby('url'):
        persons = group[group['type'] == 'PERSON']['entity'].tolist()
        orgs    = group[group['type'] == 'ORG']['entity'].tolist()
        gpes    = group[group['type'] == 'GPE']['entity'].tolist()

        added = 0

        # PERSON --memberOf--> ORG
        for p in persons:
            for o in orgs:
                if added >= MAX_RELATIONS_PER_PAGE:
                    break
                pu = entity_index.get(p)
                ou = entity_index.get(o)
                if pu and ou:
                    g.add((pu, ONT.memberOf, ou))
                    added += 1
                    relation_count += 1

        # PERSON --basedIn--> GPE
        for p in persons:
            for gpe in gpes[:3]:   # max 3 lieux par personne
                if added >= MAX_RELATIONS_PER_PAGE:
                    break
                pu   = entity_index.get(p)
                gpeu = entity_index.get(gpe)
                if pu and gpeu:
                    g.add((pu, ONT.basedIn, gpeu))
                    added += 1
                    relation_count += 1

        # ORG --locatedIn--> GPE
        for o in orgs:
            for gpe in gpes[:3]:
                if added >= MAX_RELATIONS_PER_PAGE:
                    break
                ou   = entity_index.get(o)
                gpeu = entity_index.get(gpe)
                if ou and gpeu:
                    g.add((ou, ONT.locatedIn, gpeu))
                    added += 1
                    relation_count += 1

    print(f"  Relations: {relation_count} triples inferred from co-occurrence.")


# ==============================
# MAIN
# ==============================

def main():
    import os
    os.makedirs("././kg_artifacts", exist_ok=True)

    print("\n=== STEP 1: Knowledge Graph Construction ===\n")

    # Load
    data = pd.read_csv(CSV_PATH)
    df = data.sample(n=200, random_state=42) 

    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=['entity', 'type', 'url'])
    print(f"  CSV loaded: {len(df)} rows, {df['type'].value_counts().to_dict()}")

    # Deduplicate
    df = deduplicate(df)
    print(f"  After dedup: {len(df)} unique entities")

    # Build graph
    g = Graph()
    g.bind("ex",   EX)
    g.bind("ont",  ONT)
    g.bind("rdfs", RDFS)
    g.bind("rdf",  RDF)
    g.bind("owl",  OWL)
    g.bind("xsd",  XSD)

    build_ontology(g)
    entity_index = add_entities(g, df)
    infer_relations(g, df, entity_index)

    # Stats
    print(f"\n  Total triples in graph: {len(g)}")

    # Save
    g.serialize(OUTPUT_TTL, format="turtle")
    print(f"\n✅ Step 1 complete → {OUTPUT_TTL}")
    print(f"   Classes   : 3 (Person, Organization, Place)")
    print(f"   Properties: 4 (memberOf, basedIn, locatedIn, mentionedIn)")
    print(f"   Entities  : {len(entity_index)}")
    print(f"   Triples   : {len(g)}")


if __name__ == "__main__":
    main()