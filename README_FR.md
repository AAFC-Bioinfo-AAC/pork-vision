<!-- omit in toc -->
# PORK-VISION

[![FR](https://img.shields.io/badge/lang-FR-yellow.svg)](README_FR.md)
[![EN](https://img.shields.io/badge/lang-EN-blue.svg)](README.md)

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-0A66C2?logo=&logoColor=white)](https://github.com/ultralytics/ultralytics)

---

<!-- omit in toc -->
## Table des matières

- [À propos](#à-propos)
- [Documentation](#documentation)
- [Remerciements](#remerciements)
- [Sécurité](#sécurité)
- [Licence](#licence)

---

## À propos

Les exportations de l’industrie porcine canadienne génèrent environ *5 milliards de dollars* par an. Les découpes primaires présentant des attributs de qualité recherchés, en particulier les longes, les poitrines et les épaules, se vendent à des prix élevés sur les marchés internationaux, comme le Japon. Les méthodes actuelles de mesure de la qualité du porc, tant en ligne qu’en conditions de recherche, reposent principalement sur des méthodes subjectives et des tests manuels sur la longe. Les systèmes entièrement automatisés ne sont généralement pas disponibles pour la collecte de données de qualité dans les découpes primaires ou les côtelettes de porc, et l’adoption des quelques technologies disponibles capables d’évaluer certains attributs de qualité a été limitée en raison de coûts élevés et d’exigences opérationnelles.

Nous avons développé ici un **pipeline d’analyse d’images basé sur Python** utilisant la **vision par ordinateur** et des techniques **d’apprentissage profond** pour **automatiser l’évaluation des côtelettes** de longe (emplacement de référence pour l’évaluation de la qualité du porc) en fonction des **attributs de qualité** les plus importants exigés par les acheteurs nationaux et internationaux. En utilisant une vaste **banque d’images phénotypiques porcines** et un ensemble de données générés au *Centre de recherche et de développement de Lacombe* (AAC, Lacombe, AB), le système a été développé et validé dans des conditions imitant la transformation commerciale. Il reproduit les flux de travail manuels traditionnellement effectués avec ImageJ et des macros personnalisées, rationalisant le processus tout en maintenant la compatibilité avec **les normes canadiennes de couleur et de persillage du porc**.

Le pipeline extrait des mesures quantitatives telles que : la largeur et la profondeur du muscle, l’épaisseur du gras, le pourcentage de persillage, et le score de couleur à partir d’images standardisées de côtelettes de porc. Il est conçu pour traiter efficacement de grands lots, ce qui le rend bien adapté aux applications de recherche et industrielles. Développé entièrement en Python, le système utilise des bibliothèques comme **PyTorch**, **OpenCV** et **NumPy**, et intègre :

- **Modèles d’apprentissage profond** :
  - Un **modèle de segmentation** pour isoler les régions de gras et de muscle
  - Un **modèle de détection d’objets YOLO11** pour identifier les étalons de couleur intégrés
- **Algorithmes personnalisés** :
  - Prétraitement des images et algorithmes de mesure pour l’analyse géométrique et basée sur l’intensité

---

## Documentation

Pour les détails techniques, y compris les instructions d’installation et d’utilisation, veuillez consulter le [Guide de l’utilisateur](/docs/user-guide.md).

---

## Remerciements

- **Crédits** :
  - Ce projet a été développé par une équipe multidisciplinaire de bio-informaticiens et de spécialistes en sciences de la viande au *Centre de recherche et de développement de Lacombe, Agriculture et Agroalimentaire Canada (AAC)*. Pour une liste des contributions individuelles, voir [CREDITS.md](CREDITS.md)
  - 🤖 Des modèles d’IA générative ont été utilisés pour la réalisation de ce projet, et tout le contenu généré par l’IA a été examiné, vérifié et perfectionné par l’équipe de projet afin d’en assurer l’exactitude.

- **Citation** : Pour citer ce projet, cliquez sur le bouton **`Cite this repository`** dans la barre latérale de droite.

- **Contribution** : Les contributions sont les bienvenues ! Veuillez consulter les lignes directrices dans [CONTRIBUTING.md](CONTRIBUTING.md) et vous assurer de respecter notre [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) afin de favoriser un environnement respectueux et inclusif.

- **Références** : Pour une liste des principales ressources utilisées ici, voir [REFERENCES.md](REFERENCES.md)

---

## Sécurité

⚠️ Ne publiez aucun problème de sécurité sur le répertoire public ! Veuillez les signaler comme décrit dans [SECURITY.md](SECURITY.md).

---

## Licence

Voir le fichier [LICENSE](LICENSE) pour plus de détails. Visitez [LicenseHub](https://licensehub.org/fr) ou [tl;drLegal](https://www.tldrlegal.com/) pour consulter un résumé en langage clair de cette licence.

**Droit d’auteur ©** Sa Majesté le Roi du chef du Canada, représenté par le ministre de l’Agriculture et de l’Agroalimentaire, 2025.

---
