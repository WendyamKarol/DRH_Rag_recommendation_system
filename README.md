# Système RAG de Recommandations de Formation

## 📋 Description

Ce projet implémente un système de **Retrieval-Augmented Generation (RAG)** pour générer des recommandations de formation personnalisées basées sur les évaluations des employés.  
Le système analyse les lacunes de compétences, recherche des formations pertinentes dans un corpus documentaire et génère des plans de développement adaptés.

## 🎯 Objectifs

- **Analyser** les évaluations des employés pour identifier les lacunes  
- **Rechercher** des formations pertinentes dans un corpus de 60 documents  
- **Générer** des recommandations personnalisées avec GPT-4  
- **Produire** des rapports structurés pour les RH  

## 🏗️ Architecture

```
RAG/
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuration centralisée
├── src/
│   ├── __init__.py
│   ├── models.py                 # Modèles de données
│   ├── rag_system.py             # Système principal
│   ├── embeddings/               # Gestion des vecteurs
│   ├── retrievers/               # Recherche de documents
│   ├── generators/               # Génération avec GPT-4
│   └── agents/                   # Multi-agents (Partie 2)
├── data/
│   ├── employee (1).json         # Évaluations employés
│   └── formation (1).json        # Corpus de formation
├── tests/
│   └── test_simple.py            # Tests unitaires
├── demo.py                       # Démonstration
├── main.py                       # Interface CLI
├── requirements.txt              # Dépendances
└── .env                          # Clé API OpenAI
```

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone [URL_DU_REPO]
cd RAG
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer l'API OpenAI
```bash
# Créer le fichier .env
cp .env.example .env

# Ajouter votre clé API dans .env
OPENAI_API_KEY=sk-votre-clé-ici
```

## 📦 Utilisation

### Démonstration rapide
```bash
python demo.py
```
➡ Traite un employé exemple et génère un rapport.

### Traiter un employé spécifique
```bash
python main.py single --name "Marie Dupont"   --evaluation "Besoin de formation en gestion de projet"   --score 70
```

### Traiter tous les employés
```bash
python main.py batch --input data/employee.json --output reports/
```

### Afficher les statistiques
```bash
python main.py stats
```

## 🔄 Workflow du système

1. **Analyse des compétences** → identifie les lacunes dans l'évaluation  
2. **Recherche vectorielle** → trouve les documents pertinents avec FAISS  
3. **Génération GPT-4** → crée des recommandations personnalisées  
4. **Rapport structuré** → produit un document formaté  

## 📊 Exemple de sortie

```
RAPPORT DE RECOMMANDATIONS DE FORMATION
======================================
Employé: Marie Dupont
Score: 70/100
Priorité: Moyenne

FORMATIONS RECOMMANDÉES:
1. Formation en Gestion de Projet Agile
   Type: Programme de formation
   Durée: 40 heures
   Pertinence: Directement lié aux lacunes identifiées

PLAN DE DÉVELOPPEMENT:
- Court terme: Inscription formation, pratique régulière
- Long terme: Certification, mentorat d'équipe
```

## 🧪 Tests

```bash
# Lancer les tests simples
python tests/test_simple.py

# Tests manuels
python test_setup.py
```

## 🏃‍♂️ Partie 2 : Architecture Multi-Agents

La partie 2 ajoute 3 agents spécialisés :
- **Agent Formation** → Recherche des programmes de formation  
- **Agent Pratiques** → Trouve des guides de bonnes pratiques  
- **Agent Cas** → Identifie des études de cas pertinentes  

```bash
# Démo multi-agents
python demo_part2.py
```

## 💰 Coûts estimés

- **Par employé** : ~0.03-0.05€ (dépend de la longueur)
- **50 employés** : ~2-3€
- **Indexation initiale** : ~0.50€ (une seule fois)

## ⚙️ Configuration

Fichier `config/config.py` :
- Modèle GPT : `gpt-4`  
- Embeddings : `text-embedding-ada-002`  
- Chunk size : 500 caractères  
- Top K : 5 documents  

## 🛠️ Technologies utilisées

- **Python 3.8+**  
- **LangChain** → Framework RAG  
- **OpenAI API** → GPT-4 et embeddings  
- **FAISS** → Recherche vectorielle  
- **Rich** → Interface console  

## 📝 Notes importantes

- La première exécution crée l'index FAISS (~2 min)  
- Les exécutions suivantes utilisent l'index en cache  
- Chaque recommandation prend ~20-30 secondes  
- Les données sont stockées localement (**RGPD compliant**)  

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository  
2. Créer une branche (`git checkout -b feature/amelioration`)  
3. Commit (`git commit -am 'Ajout fonctionnalité'`)  
4. Push (`git push origin feature/amelioration`)  
5. Créer une Pull Request  

## 📞 Support

Pour toute question :
- Créer une issue sur GitHub  
- Contact : w.karolnaze@gmail.com  

---

✅ **Développé pour Devoteam - Test Technique**
