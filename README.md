# Drh_recommendation_RAG_System

### Arborescence du projet 

rag/
├── config/
│   └── config.py                     # Paramètres globaux (modèles, chemins, seuils, mapping, etc.)
│
├── data/
│   ├── employee.json                 # Évaluations RH à analyser
│   └── formation.json                # Corpus de documents de formation (base vectorielle)
│
├── src/
│   ├── models.py                     # Modèles Pydantic pour structurer les données RH, documents, recommandations
│   ├── rag_system.py                 # Orchestrateur principal (pipeline analyse → retrieval → génération)
│
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py                   # Interface abstraite pour moteurs d’embedding
│   │   └── faiss_store.py            # Implémentation FAISS (indexation et recherche vectorielle)
│
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base.py                   # Interface des générateurs
│   │   └── generator.py              # Générateur OpenAI (prompt → réponse JSON formatée)
│
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── base.py                   # Interface de récupérateur
│   │   └── retriever.py              # Implémentation simple : utilise FAISS et top-k
│
├── tests/
│   └── test_generators.py            # Test unitaire de la génération (mock OpenAI)
│
├── main.py                           # Script CLI principal (traitement d’une ou plusieurs évaluations)
├── requirements.txt                  # Dépendances du projet
└── .env                              # Fichier d’environnement (clé OpenAI)


## Exécution

Avant d'exécuter le programme, assurez-vous que :

- Le fichier `.env` contient votre clé API OpenAI (`OPENAI_API_KEY=...`)
- Les dépendances sont installées :

```bash
pip install -r requirements.txt
```

- Les fichiers suivants existent dans le dossier `data/` :
  - `employee.json`
  - `formation.json`
  - Les fichiers FAISS (`faiss_index_*.index`, `faiss_index_*.docs`) correspondant aux différents types de documents

---

### Mode 1 – Générer un rapport pour un seul employé

```bash
python main.py single --name "Julie Lambert"
```

> Le nom doit correspondre **exactement** à celui défini dans `employee.json`

---

### Mode 2 – Générer des rapports pour tous les employés

```bash
python main.py batch
```

> Tous les rapports sont automatiquement sauvegardés dans le dossier `reports/`, un fichier `.txt` par employé.
