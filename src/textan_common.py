#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Ce fichier contient la classe TextAnCommon, utilisée pour la résolution de la problématique.
    Ce code ne devrait pas être modifié, il contient des méthodes utiles qui sont utilisées dans le gabarit de solution

    Les méthodes apparaissant dans ce fichier définissent des fonctionnalités de base qui sont utilisées dans
    la classe TextAn.

    La classe TextAn est invoquée par la classe TestTextAn (contenue dans test_textan.py) :

        - Tous les arguments requis sont présents et accessibles dans args (dans le fichier test_textan.py)
        - Note : vous pouvez tester votre code en utilisant les commandes :
            + "python test_textan.py"
            + "python test_textan.py -h" (donne la liste des arguments possibles)
            + "python test_textan.py -v" (mode "verbose", qui indique les valeurs de tous les arguments)

    Copyright 2018-2026 F. Mailhot
"""
import os
import ntpath
import math

from handle_unicode_common import HandleUnicodeCommon


class TextAnCommon(HandleUnicodeCommon):
    # Le code qui suit est fourni pour vous faciliter la vie.  Il n'a pas à être modifié

    # Signes de ponctuation à traiter comme des mots (compléter la liste qui ne comprend que "!" au départ)
    # Modifier cette liste dans votre code (textan_CIP1_CIP2.py).
    PONC = ["!", ";"]

    @staticmethod
    def get_empty_ngram(size: int) -> list[str]:
        """Retourne un ngramme vide de la taille indiquée (liste contenant des chaînes de caractères vides)

        Args :
            size (int) : le nombre de mots vides dans la liste ngramme

        Returns :
            ngram (liste) : La liste de mots vides

        Copyright 2024-2026 F. Mailhot
        """
        ngram = [""] * size
        return ngram

    @staticmethod
    def subtract_dict(dict1: dict, dict2: dict) -> dict:
        """Calcule la différence entre les dimensions de deux vecteurs
            Note : Toutes les dimensions (clés) du deuxième vecteur (dictionnaire)
                    sont retirées du premier vecteur (dictionnaire)

        Args :
            dict1 (dict) : le premier vecteur
            dict2 (dict) : le deuxième vecteur

        Returns :
            sub_dict (dict) : Le vecteur (dictionnaire) résultant

        Copyright 2026 F. Mailhot
        """
        sub_dict = {k: v for k, v in dict1.items() if k not in dict2}
        return sub_dict

    def get_aut_files(self, auteur):
        """Obtient la liste des fichiers (avec le chemin complet) des oeuvres d'un auteur

        Args :
            auteur (string) : le nom de l'auteur dont on veut obtenir la liste des oeuvres

        Returns :
            oeuvres (Liste[string]) : liste des oeuvres (avec le chemin complet pour y accéder)

        Copyright 2024-2026 F. Mailhot
        """
        return self.oeuvres_auteurs[auteur]

    def set_aut_files(self, auteur) -> None:
        """Met en mémoire la liste des fichiers (avec le chemin complet) des oeuvres d'un auteur

        Args :
            auteur (string) : le nom de l'auteur dont on veut obtenir la liste des oeuvres

        Returns :
            (void) : La liste des oeuvres de l'auteur est ajoutée dans le dictionnaire self.oeuvres_auteurs

        Copyright 2024-2026 F. Mailhot
        """
        # https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
        oeuvres = [f.path for f in os.scandir(os.path.join(self.rep_aut, auteur))
                   if f.is_file() and f.name.endswith('.txt')]
        self.oeuvres_auteurs[auteur] = oeuvres
        return

    def set_auteurs(self):
        """Obtient la liste des auteurs, à partir du répertoire qui les contient tous

        Note : le champ self.rep_aut doit être prédéfini :
            - Par défaut, il contient le répertoire d'exécution du script
            - Peut être redéfini par la méthode set_aut_dir

        Returns :
            (void) : ne fait qu'obtenir la liste des répertoires d'auteurs et modifier la liste self.auteurs

        Copyright 2018-2026 F. Mailhot
        """
        # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
        full_path_auteurs = [f.path for f in os.scandir(self.rep_aut) if f.is_dir()]
        self.ngrams_auteurs = {}
        for auteur_path in full_path_auteurs:
            auteur = TextAnCommon.normalize_string(ntpath.basename(auteur_path))
            self.auteurs.append(auteur)
            self.ngrams_auteurs[auteur] = {}
            self.set_aut_files(auteur)
        return

    def set_aut_dir(self, aut_dir):
        """Définit le nom du répertoire qui contient l'ensemble des répertoires d'auteurs

        Note : L'appel à cette méthode
            - extrait la liste des répertoires d'auteurs et les ajoute à self.auteurs
            - extrait la liste des oeuvres de chaque auteur et l'ajoute à self.oeuvres_auteurs[auteur]

        Args :
            aut_dir (string) : Nom du répertoire (peut être absolu ou bien relatif au répertoire d'exécution)

        Returns :
            (void) : ne fait que définir le nom du répertoire qui contient les répertoires d'auteurs

        Copyright 2018-2026 F. Mailhot
        """
        cwd = os.getcwd()
        if os.path.isabs(aut_dir):
            self.rep_aut = aut_dir
        else:
            self.rep_aut = os.path.join(cwd, aut_dir)

        self.rep_aut = os.path.normpath(self.rep_aut)
        self.set_auteurs()

        return

    def set_ngram_size(self, ngram_size):
        """Indique que l'analyse et la génération de texte se fera avec des n-grammes de taille ngram

        Args :
            ngram (int) : Indique la taille des n-grammes (1, 2, 3, ...)

        Returns :
            (void) : ne fait que mettre à jour le champ ngram

        Copyright 2018-2026 F. Mailhot
        """
        self.ngram_size = ngram_size
        return

    def get_ngram_occurrence(self, auteur: str, ngram: str) -> int:
        """Retourne le nombre d'occurrences du n-gramme pour cet auteur

        Args :
            auteur (string) : le nom de l'auteur
            ngram (objet de type ngram) : le n-gramme dont on désire la fréquence

        Returns :
            (int) : retourne ne nombre d'occurrences du n-gramme pour l'auteur donné

        Copyright 2024-2026 F. Mailhot
        """
        return 0

    def get_total_occurrences(self, auteur: str) -> int:
        """Retourne le nombre total d'occurrences de n-grammes pour cet auteur

        Args :
            auteur (string) : le nom de l'auteur

        Returns :
            (int) : retourne ne nombre total d'occurrences pour l'auteur donné

        Copyright 2024-2026 F. Mailhot
        """
        return 1

    def normalize_vector(self, dict_de_ngrams: dict) -> dict:
        """Normalize le vecteur (dictionnaire), en divisant chaque occurrence par la taille totale

        Args :
            dict_de_ngrams (dict) : le vecteur de n-grammes (dict) en question

        Returns :
            (dict) : Une nouvelle version normalisée du dictionnaire est retournée

        Copyright 2024-2026 F. Mailhot
        """
        norm_dict = {}
        return norm_dict

    def normalize_ngrams_auteurs(self) -> None:
        """Normalise tous les vecteurs de n-grammes de chacun des auteurs

            Args :
                (void) : Tout se trouve dans l'objet
                    (la liste d'auteurs, ainsi que les vecteurs de n-grammes de chacun)

            Returns :
                (void) : Le tableau self.normalized_ngrams_auteurs est créé avec les vecteurs de n-grammes normalisés

            Copyright 2026 F. Mailhot
            """
        for auteur in self.auteurs:
            self.normalized_ngrams_auteurs[auteur] = self.normalize_vector(self.ngrams_auteurs[auteur])
        return

    def __init__(self) -> None:
        """Initialize l'objet de type TextAn lorsqu'il est créé

        Args :
            (void) : Utilise simplement les informations fournies dans l'objet Textan_config

        Returns :
            (void) : ne fait qu'initialiser l'objet de type TextAn

        Copyright 2018-2026 F. Mailhot
        """

        # Initialisation des champs nécessaires aux fonctions fournies
        self.cip = ""
        self.rep_aut = os.getcwd()
        self.auteurs = []
        self.ngram_size = 1
        self.ngrams_auteurs = {}
        self.normalized_ngrams_auteurs = {}
        self.oeuvres_auteurs = {}
        return
