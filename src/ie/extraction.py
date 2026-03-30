import httpx
import trafilatura
import spacy
import pandas as pd
import json

# Configuration
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Olympic_Games",
    "https://en.wikipedia.org/wiki/Winter_sports",
    "https://en.wikipedia.org/wiki/Athlete",
    "https://en.wikipedia.org/wiki/World_championship",
    "https://en.wikipedia.org/wiki/Ancient_Olympic_Games",
    "https://en.wikipedia.org/wiki/Paralympic_Games",
    "https://en.wikipedia.org/wiki/Olympic_symbols#Flag",
]   

# Chargement du modèle
nlp = spacy.load("en_core_web_trf")

def is_useful(text, min_words=500):
    return len(text.split()) >= min_words

def run_pipeline():
    crawler_results = []
    extracted_data = []

    print("--- Phase 1: Crawling & Cleaning ---")
    for url in SEED_URLS:
        try:
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded)
            
            if content and is_useful(content):
                entry = {"url": url, "text": content}
                crawler_results.append(entry)
                print(f"Succès : {url}")
                
                # Phase 2: NER (Reconnaissance d'entités)
                doc = nlp(content)
                for ent in doc.ents:

                    if ent.label_ in ["PERSON", "ORG", "GPE"]:
                        clean_text = ent.text.replace("\n", " ").strip()
                        
                        extracted_data.append({
                            "entity": clean_text,
                            "type": ent.label_,
                            "url": url
                        })
        except Exception as e:
            print(f"Erreur sur {url}: {e}")

    with open('././data/crawler_output.jsonl', 'w', encoding='utf-8') as f:
        for item in crawler_results:
            f.write(json.dumps(item) + '\n')

    df = pd.DataFrame(extracted_data)
    
    # 1. Supprimer les doublons 
    df = df.drop_duplicates()
    
    df = df[df['entity'].str.len() > 2]

    # Export du CSV final
    df.to_csv('././data/extracted_knowledge.csv', index=False, encoding='utf-8')
    print(f"\n--- Terminé : {len(df)} entités uniques extraites (Dates exclues) ---")

if __name__ == "__main__":
    run_pipeline()