#  Système d’évitement d’obstacles en temps réel avec YOLOv8 (IA embarquée)

## 📌 Description

Ce projet consiste en la conception et l’implémentation d’un **système d’évitement d’obstacles en temps réel** basé sur l’algorithme de détection d’objets **YOLOv8**, déployé sur un **Raspberry Pi 4**.

 

---

## 🎯 Objectifs

* Réaliser une détection d’objets en temps réel avec YOLOv8
* Estimer la proximité des obstacles à partir de la vision
* Contrôler un robot mobile en fonction des détections

---
 

 

## 🧠 Principe de fonctionnement (logique d’évitement)

La proximité des obstacles est estimée à partir de la taille de l’objet détecté :

* 📏 Bounding box large → objet **proche**
* 📏 Bounding box petite → objet **éloigné**

 

* Si largeur > seuil → obstacle détecté → réaction du robot (arrêt / évitement)
* Sinon → déplacement normal

---

 

## 🔌 Plateforme embarquée

* Raspberry Pi 4
* Module caméra
* Robot mobile

---

## 🚀 Étapes de réalisation

### 1. Mise en place de YOLOv8

* Installation et configuration sur Raspberry Pi
* Tests de détection en temps réel

### 2. Intégration de la caméra

* Acquisition du flux vidéo
* Validation du fonctionnement

### 3. Intégration du robot

* Tests de déplacement
* Liaison avec la logique de détection

### 4. Système complet

* Couplage détection + contrôle
* Tests en conditions réelles

### 5. Interface web

* Développement d’un serveur Flask
* Streaming vidéo en temps réel

### 6. Automatisation

* Configuration d’un service systemd
* Lancement automatique au démarrage

- 

 

 

 

---

 