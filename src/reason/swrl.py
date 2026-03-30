from owlready2 import *
import os

# owlready2.JAVA_EXE = r"C:\Program Files\Java\jdk-22\bin\java.exe"
# Configuration des chemins
# FAMILY_OWL = "../../data/family.owl"
MY_ONTO    = "././kg_artifacts/ontology.owl"
# On enregistre le résultat dans un nouveau fichier pour ne pas écraser l'original
OUTPUT_REASONING = "././kg_artifacts/graph_enriched.owl"



def reason_on_my_kb():
    print("\n--- Exercice 2 : Graphe Olympique (Règle personnalisée) ---")
    
    # Charger ton ontologie OWL/XML
    onto = get_ontology(MY_ONTO).load()
    
    with onto:
        # Déclarer les classes/propriétés pour qu'OWLReady2 les lie au fichier
        # On utilise les noms EXACTS de ton fichier XML
        class Person(Thing): pass
        class Organization(Thing): pass
        class Place(Thing): pass
        
        # On crée une nouvelle classe pour le résultat du raisonnement
        class VIP_Entity(Thing): pass 

        # Propriétés existantes dans ton XML
        class memberOf(ObjectProperty): pass
        class locatedIn(ObjectProperty): pass

        # --- TA RÈGLE PERSONNALISÉE ---
        # Logique : Si une Personne est membre d'une Organisation 
        # ET que cette Organisation est située dans un Lieu (Place),
        # Alors la Personne est une VIP_Entity (Entité localisée/affiliée).
        rule = Imp()
        rule.set_as_rule("Person(?p), memberOf(?p, ?o), Organization(?o), locatedIn(?o, ?l) -> VIP_Entity(?p)")
        
        print("Lancement du raisonneur Pellet (Inférence en cours)...")
        # infer_property_values permet de créer les nouveaux liens
        owlready2.reasoning.JAVA_MEMORY = 1000
        sync_reasoner_pellet(infer_property_values=True)

    # Sauvegarde
    onto.save(file=OUTPUT_REASONING, format="rdfxml")
    print(f"✅ Raisonnement réussi ! Graphe enrichi : {OUTPUT_REASONING}")

if __name__ == "__main__":
    # Note : Assure-toi d'avoir Java installé (Pellet en a besoin)
    try:
        reason_on_my_kb()
    except Exception as e:
        print(f"❌ Erreur lors du raisonnement : {e}")