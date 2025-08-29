# PORK-VISION

![Licence](https://img.shields.io/badge/License-GPLv3-blue.svg)

<!-- omit in toc -->
## Table des matières

- [PORK-VISION](#pork-vision)
  - [À propos](#à-propos)
  - [Crédits](#crédits)
  - [Contribution](#contribution)
  - [Licence](#licence)
  - [Références](#références)
  - [Citation](#citation)

---

## À propos

Les exportations de l'industrie porcine canadienne génèrent 5 milliards de dollars par an. Les morceaux primaires avec des attributs de qualité désirables, en particulier les longes, poitrines et épaules, sont vendus à des prix élevés sur les marchés internationaux, tels que le Japon. Les méthodes actuelles utilisées pour mesurer la qualité du porc, à la fois en ligne et dans des conditions de recherche, reposent principalement sur des méthodes subjectives et des tests manuels sur la longe. Les systèmes entièrement automatisés ne sont généralement pas disponibles pour la collecte de données de qualité sur les morceaux primaires ou les côtelettes de porc, et l'adoption des quelques technologies disponibles capables d'évaluer certains attributs de qualité a été limitée en raison de coûts élevés et d'exigences opérationnelles.

Nous avons développé ici un pipeline d'analyse d'images basé sur Python utilisant la vision par ordinateur et des techniques d'apprentissage profond pour automatiser l'évaluation des côtelettes de longe (endroit de référence pour l'évaluation de la qualité du porc) en fonction des attributs de qualité les plus importants requis par les acheteurs nationaux et internationaux. En utilisant une vaste banque d'images phénotypiques de porc et un jeu de données généré au Centre de recherche et de développement de Lacombe d'AAC (Lacombe, AB), le système a été développé et validé dans des conditions simulant la transformation commerciale. Il reproduit les flux de travail manuels traditionnellement réalisés avec ImageJ et des macros personnalisées, rationalisant le processus tout en maintenant la compatibilité avec les normes canadiennes de couleur et de persillage du porc.

Le pipeline extrait des mesures quantitatives telles que la largeur et la profondeur du muscle, la profondeur de la graisse, le pourcentage de persillage et l’indice de couleur à partir d’images standardisées de côtelettes de porc. Il est conçu pour traiter efficacement de grands lots, ce qui le rend adapté aussi bien à la recherche qu’aux applications industrielles. Entièrement développé en Python, le système exploite des bibliothèques telles que PyTorch, OpenCV et NumPy, et intègre :

- Modèles d’apprentissage profond :
  - Un modèle de segmentation pour isoler les régions de graisse et de muscle
  - Un modèle de détection d’objets YOLOv11 pour identifier les étalons de couleur intégrés
- Algorithmes personnalisés :
  - Prétraitement des images et algorithmes de mesure pour l’analyse géométrique et basée sur l’intensité
  
Pour les détails techniques, l’installation et l’utilisation, voir le [Guide d’utilisation](./docs/user-guide.md).

---

## Crédits

Développé par une équipe pluridisciplinaire de bio-informaticiens et d’experts en sciences de la viande au Centre de recherche et de développement de Lacombe, Agriculture et Agroalimentaire Canada. Pour les contributions individuelles, voir le fichier [CREDITS](CREDITS.md).

---

## Contribution

Si vous souhaitez contribuer à ce projet, veuillez consulter les lignes directrices dans [CONTRIBUTING.md](CONTRIBUTING.md) et vous assurer de respecter notre [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## Licence

Ce projet est distribué sous la licence GPLv3. Pour plus de détails et d’informations sur les droits d’auteur, voir le fichier [LICENSE](LICENSE).

---

## Références

Les références aux outils et logiciels utilisés ici se trouvent dans le fichier [CITATIONS.md](CITATIONS.md).

---

## Citation

Si vous utilisez ce projet dans vos travaux, veuillez le citer en utilisant le fichier [CITATION.cff](CITATION.cff).
