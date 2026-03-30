from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal
import os

# Chemins : On passe en .owl
INPUT_GRAPH = "././kg_artifacts/graph.ttl"
OUTPUT_ONTO = "././kg_artifacts/ontology.owl"

def generate_ontology():
    g_input = Graph()
    # On charge ton graphe de données (Turtle)
    g_input.parse(INPUT_GRAPH, format="turtle")
    
    onto = Graph()
    # Espace de noms pour ton ontologie
    EX = Namespace("http://myolympicgraph.org/ontology#") 
    onto.bind("ex", EX)
    onto.bind("owl", OWL)
    onto.bind("rdfs", RDFS)

    print("Extraction des classes et propriétés pour format OWL/XML...")

    # 1. Extraire les Classes
    for s, p, o in g_input.triples((None, RDF.type, None)):
        if "http" in str(o):
            onto.add((o, RDF.type, OWL.Class))
            # On génère un label lisible à partir de l'URI
            label_val = str(o).split('/')[-1].split('#')[-1]
            onto.add((o, RDFS.label, Literal(label_val, lang="en")))

    # 2. Extraire les Propriétés (ObjectProperties)
    for s, p, o in g_input.triples((None, None, None)):
        if p != RDF.type:
            onto.add((p, RDF.type, OWL.ObjectProperty))
            
            # Essayer de deviner Domain et Range pour aider le raisonneur
            s_type = g_input.value(s, RDF.type)
            o_type = g_input.value(o, RDF.type)
            if s_type: onto.add((p, RDFS.domain, s_type))
            if o_type: onto.add((p, RDFS.range, o_type))

    # SAUVEGARDE AU FORMAT OWL (XML)
    # 'pretty-xml' est le format standard pour les fichiers .owl
    onto.serialize(destination=OUTPUT_ONTO, format="pretty-xml")
    
    print(f"✅ Ontologie générée avec succès en format OWL/XML : {OUTPUT_ONTO}")

if __name__ == "__main__":
    generate_ontology()