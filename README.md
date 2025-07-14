# Simulation Monte Carlo & Analyse Technique

Ce projet permet de simuler des trajectoires de prix d’actifs financiers à partir de données historiques téléchargées via **yfinance**, en combinant des indicateurs techniques (EMA20, RSI) et une simulation **Monte Carlo** basée sur un modèle géométrique de marche brownienne.

📈 Les données historiques sont récupérées automatiquement pour un ticker et une période donnés. Les indicateurs techniques (moyenne mobile exponentielle 20 jours, RSI 14 jours) sont calculés sur ces données.

🎲 La simulation Monte Carlo génère plusieurs trajectoires futures de prix, à partir des paramètres statistiques (drift **mu**, volatilité **sigma**) estimés sur les rendements logarithmiques historiques. Les trajectoires simulées sont affichées en prolongement temporel à la suite des données réelles, pour visualiser les scénarios possibles.

📊 Le projet est structuré autour d’une classe `Stock` qui regroupe les fonctionnalités suivantes :  
- Téléchargement et préparation des données  
- Calcul des indicateurs techniques  
- Estimation des paramètres pour la simulation Monte Carlo  
- Sauvegarde des paramètres au format JSON  
- Simulation des trajectoires  
- Visualisation combinée des prix réels, indicateurs, et trajectoires simulées sur un seul graphique  

---

## Structure du projet

- `Stock` : classe principale avec méthodes pour l’ensemble des étapes  
- Récupération automatique des données via `yfinance`  
- Calcul des indicateurs EMA20 et RSI  
- Estimation de **mu**, **sigma**, **S0**, et horizon **T** à partir des données  
- Simulation Monte Carlo des trajectoires de prix futures  
- Affichage unique regroupant prix historiques, indicateurs, et projections Monte Carlo  

---

## Utilisation

1. Instancier un objet `Stock` avec un ticker et une période  
2. Appeler successivement :  
   - `telecharger_donnees()`  
   - `calculer_indicateurs()`  
   - `calculer_parametres()`  
   - `sauvegarder_parametres()` (optionnel)  
   - `afficher_graphique_prolonge()` pour visualiser les données et simulations  

---

## Objectif

Offrir un outil simple, pédagogique et modulaire permettant de :  
- Analyser les données financières historiques d’un actif  
- Calculer des indicateurs techniques courants  
- Simuler et visualiser des trajectoires de prix futures via Monte Carlo  
- Intégrer analyse technique et modélisation stochastique dans un workflow fluide  

---
