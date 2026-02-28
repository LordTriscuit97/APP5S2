#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Ce fichier contient la classe TextAn, à utiliser pour résoudre la problématique.
    C'est un gabarit pour l'application de traitement des fréquences de mots dans les oeuvres d'auteurs divers.

    Les méthodes apparaissant dans ce fichier définissent une API qui est utilisée par l'application
    de test test_textan.py
    Les paramètres d'entrée et de sortie (Application Programming Interface, API) sont définis,
    mais le code est à écrire au complet.
    Vous pouvez ajouter toutes les méthodes et toutes les variables nécessaires au bon fonctionnement du système

    La classe TextAn est invoquée par la classe TestTextAn (contenue dans test_textan.py) :

        - Tous les arguments requis sont présents et accessibles dans args (dans le fichier test_textan.py)
        - Les arguments proviennent :
            - soit du fichier de configuration test_textan_config.yml,
            - soit de la ligne de commande
        - Note : vous pouvez tester votre code en utilisant les commandes :
            + "python test_textan.py"
            + "python test_textan.py -h" (donne la liste des arguments possibles)
            + "python test_textan.py --v" (mode "verbose", qui indique les valeurs de tous les arguments)
        - Note (2) : vous pouvez modifier le fichier test_textan_config.yml :
            - Vous le trouverez dans le répertoire de travail (Problematique/data)
            - Les mêmes options existent dans le fichier test_textan_config.yml et en ligne de commande
            - Les paramètres passés en ligne de commande ont priorité sur ceux définis dans le fichier de configuration
        - Note (3) : pour exécuter le harnais de test test_textan.py :
            - Il est nécessaire que le répertoire de travail soit :
                - Problematique/data
            - Dans PyCharm, vous pouvez définir à la fois:
                - le répertoire de travail (Problematique/data)
                - le script à utiliser (Problematique/src/test_textan.py)
                - les paramètres en ligne de commande (le fichier de configuration donne les valeurs par défaut)

    Copyright 2018-2026 F. Mailhot
"""
import io
from typing import TextIO
# import math  # Au besoin, retirer le commentaire de cette ligne
# import random # Au besoin, retirer le commentaire de cette ligne
from textan_common import TextAnCommon


class TextAn(TextAnCommon):
    """Classe à utiliser pour coder la solution à la problématique :

        - La classe héritée TextAnCommon contient certaines fonctions de base pour faciliter le travail :
            - recherche des auteurs
            - ouverture des répertoires
            - obtention de la liste des oeuvres d'un auteur (get_aut_files(auteur))
            - et autres (voir la classe TextAnCommon pour plus d'information)
        - Les interfaces du code à développer sont présentes, mais tout le code est à écrire
        - En particulier, il faut compléter les fonctions suivantes :
            - add_dict(dict1, dict2)
            - analyze()
            - dot_product_dict (dict1, dict2)
            - find_author (oeuvre)
            - gen_text_dict(auteur_dict, taille, to_file)
            - get_kth_element (auteur, k)
            - get_ngram_occurrence (auteur, ngram)
            - get_total_occurrences (auteur)
            - get_vector_size(dict)
            - normalize_vector(dict)


    Copyright 2018-2026 F. Mailhot
    """

    # Signes de ponctuation à traiter comme des mots (compléter cette liste incomplète)
    PONC = ["!", ";"]

    # Ajouter les structures de données et les fonctions nécessaires à l'analyse des textes,
    # la production de textes aléatoires, la détection d'oeuvres inconnues,
    # l'identification des k-ièmes mots les plus fréquents.
    #
    # Les méthodes qui suivent doivent toutes être complétées pour que le système soit opérationnel
    # et que le harnais de test (test_textan.py) puisse exécuter tous les tests requis.
    #
    #  Note : Voir la documentation ReadTheDocs
    #

    @staticmethod
    def get_vector_size(dict_de_ngrams: dict) -> float:
        """Calcule la longueur (norme) du vecteur (dictionnaire) de ngrams contenus dans un dictionnaire

        Args :
            dict_de_ngrams (dict) : le vecteur de ngrams (dict) en question

        Returns :
            taille (float) : La norme du vecteur (dict) est retournée

        Copyright 2024-2026 F. Mailhot
        """
        # Les lignes qui suivent ne servent qu'à éliminer un avertissement.
        # Il faut les retirer et les remplacer par du code fonctionnel
        print(dict_de_ngrams)
        taille = 1.0
        return taille

    def normalize_vector(self, dict_de_ngrams: dict) -> dict:
        """Normalize le vecteur (dictionnaire), en divisant chaque occurrence par la taille totale

        Args :
            dict_de_ngrams (dict) : le vecteur de n-grammes (dict) en question

        Returns :
            (dict) : Une nouvelle version normalisée du dictionnaire est retournée

        Copyright 2024-2026 F. Mailhot
        """
        # Les lignes qui suivent ne servent qu'à éliminer un avertissement.
        # Il faut les retirer et les remplacer par du code fonctionnel
        print(self.ngram_size)
        print(dict_de_ngrams)
        norm_dict = {}
        return norm_dict

    @staticmethod
    def add_dict(dict1: dict, dict2: dict) -> dict:
        """Additionne deux vecteurs représentés par des dictionnaires
        Note : le vecteur de retour n'est PAS NORMALISÉ

        Args :
            dict1 (dict) : le premier vecteur

            dict2 (dict) : le deuxième vecteur

        Returns :
            sum_dict (dict) : La somme des deux vecteurs passés en paramètre

        Copyright 2026 F. Mailhot
        """
        # Les lignes qui suivent ne servent qu'à éliminer un avertissement.
        # Il faut les retirer et les remplacer par du code fonctionnel
        sum_dict = {}
        print(dict1)
        print(dict2)
        return sum_dict

    @staticmethod
    def dot_product_dict(dict1: dict, dict2: dict) -> float:
        """Calcule le produit scalaire de deux vecteurs représentés par des dictionnaires
            Note : ce produit scalaire n'est PAS normalisé

        Args :
            dict1 (dict) : le premier vecteur
            dict2 (dict) : le deuxième vecteur

        Returns :
            dot_product (float) : Le produit scalaire normalisé de deux vecteurs

        Copyright 2024-2026 F. Mailhot
        """

        # Les lignes qui suivent ne servent qu'à éliminer un avertissement.
        # Il faut les retirer et les remplacer par du code fonctionnel
        print("\t", dict1, dict2)
        dot_product = 1.0
        return dot_product

    def find_author(self, oeuvre: str) -> list[tuple[str, float]]:
        """Après analyse des textes d'auteurs connus, retourner la liste d'auteurs
            et le niveau de proximité (un nombre entre 0 et 1) de l'oeuvre inconnue
            avec les écrits de chacun d'entre eux

        Args :
            oeuvre (str) : Nom du fichier contenant l'oeuvre d'un auteur inconnu

        Returns :
            resultats (Liste[(string, float)]) : Liste de tuples (auteur, niveau de proximité),
            où la proximité est un nombre entre 0 et 1)
        """
        # La ligne suivante ne sert qu'à éliminer un avertissement.
        # Il faut la retirer lorsque le code est complété
        print("\tAuteurs: ", self.auteurs, "\n\tOeuvre: ", oeuvre)

        # Exemple du format des sorties
        resultats = [
            ("Premier_auteur", 0.1234),
            ("Deuxième_auteur", 0.1123),
        ]

        # Exemple de lecture du fichier oeuvre une ligne à la fois.  Modifier ou remplacer ce code par le vôtre.
        fichier_oeuvre = open(oeuvre, "r", encoding="utf8")
        lignes = fichier_oeuvre.readlines()
        plus_grande_ligne = ""
        for ligne in lignes:
            if len(ligne) > len(plus_grande_ligne):
                plus_grande_ligne = ligne
        print("\tPlus grande ligne: ", plus_grande_ligne.strip())

        # Ajouter votre code pour déterminer la proximité du fichier passé en paramètre avec chacun des auteurs
        # Retourner la liste des auteurs, chacun avec sa proximité au fichier inconnu
        # Plus la proximité est grande, plus proche l'oeuvre inconnue est des autres écrits d'un auteur
        #   Le produit scalaire entre le vecteur représentant les oeuvres d'un auteur
        #       et celui associé au texte inconnu pourrait s'avérer intéressant...
        #   Le résultat du produit scalaire doit ensuite être normalisé
        #   avec la taille du vecteur associé au texte inconnu :
        #   proximité = (A dot product B) / (|A| |B|)   où A est le vecteur du texte inconnu et B est celui d'un auteur,
        #           "dot product" est le produit scalaire, et |X| est la norme (longueur) du vecteur X.
        #   À la fin, le produit scalaire normalisé représente le cosinus de l'angle (oeuvre/auteur)

        return resultats

    def get_ngram_occurrence(self, auteur: str, ngram) -> int:
        """Retourne le nombre d'occurrences du n-gramme pour cet auteur

        Args :
            auteur (string) : le nom de l'auteur

            ngram (objet du type utilisé pour un ngram) : le n-gramme dont on désire la fréquence

        Returns :
            (int) : retourne le nombre d'occurrences du n-gramme pour l'auteur donné

        Copyright 2024-2026 F. Mailhot
        """
        # Les lignes qui suivent ne servent qu'à éliminer un avertissement.
        # Il faut les retirer et les remplacer par du code fonctionnel
        print("\t", self.ngram_size, auteur, ngram)
        return 0

    def get_total_occurrences(self, auteur: str) -> int:
        """Retourne le nombre total d'occurrences de n-grammes pour cet auteur
            - Représente le total de n-grammes pour l'ensemble des oeuvres de cet auteur
            - Ce nombre est différent de la norme du vecteur :
                - il s'agit seulement du total d'occurrences de l'ensemble des ngrammes
                - Le calcul doit donner la somme des valeurs, et non la racine carrée de la somme des carrés des valeurs

        Args :
            auteur (string) : le nom de l'auteur

        Returns :
            (int) : retourne le nombre total d'occurrences pour l'auteur donné

        Copyright 2024-2026 F. Mailhot
        """
        # Les lignes qui suivent ne servent qu'à éliminer un avertissement.
        # Il faut les retirer et les remplacer par du code fonctionnel
        print("\t", self.ngram_size, auteur)
        return 1

    def gen_text_dict(self, auteur_dict: dict, taille: int, to_file: TextIO) -> None:
        """Après analyse des textes d'auteurs connus, produire un texte selon des statistiques d'un dictionnaire

        Args :
            auteur_dict (dict) : Dictionnaire à utiliser (soit d'un auteur, ou d'un amalgame d'auteurs)
            taille (int) : Taille du texte à générer
            to_file (TextIO) : Pointeur vers le fichier à créer.

        Returns :
            (void) : ne retourne rien, le texte produit doit être écrit dans le fichier fourni,
                    comprenant à la fin une série de "taille" mots séparés par des espaces
        """
        # Les lignes qui suivent ne servent qu'à éliminer un avertissement.
        # Il faut les retirer et les remplacer par du code fonctionnel
        print(self.ngram_size, auteur_dict, taille, file=to_file)
        return

    def get_kth_element(self, auteur: str, k: int) -> list[list[str]]:
        """Après analyse des textes d'auteurs connus, retourner la liste des k-ièmes plus fréquents
         n-grammes de l'auteur indiqué
         Note : il peut y avoir plus d'un n-gramme avec le même nombre d'occurrences.

        Args :
            auteur (str) : Nom de l'auteur à utiliser
            k (int) : Indice du n-gramme à retourner

        Returns :
            ngram (List[Liste[string]]) : Liste de listes de mots composant le n-gramme recherché
            (il est possible qu'il y ait plus d'un n-gramme au même rang)
        """
        # Les lignes suivantes ne servent qu'à éliminer un avertissement.
        # Il faut les retirer lorsque le code est complété
        print("\t", self.auteurs, auteur, k)
        ngram = [["un", "roman"], ["le", "lac"], ["code", "est"]]  # Exemple du format de sortie pour trois bigrammes
        return ngram

    def get_text_size(self, oeuvre: str) -> int:
        """Calcule le nombre de mots dans une oeuvre

        Args :
            oeuvre (str) : Nom du fichier contenant les mots

        Returns :
            word_number (int) : Nombre de mots dans le fichier
            Note: les signes de ponctuation sont considérés des mots
        """
        word_number = 0
        print(self.auteurs, oeuvre)
        return word_number

    def compute_ngram_stats(self, dict_de_ngrams: dict, oeuvre: str) -> None:
        """Calcule la fréquence des n-grammes dans une oeuvre, avec les n-grammes comme clé dans
        le dictionnaire de ngrams dict_de_ngrams, le nombre d'occurrences étant la valeur

        Args :
            dict_de_ngrams (dict) : Dictionnaire de n-grammes à utiliser
            oeuvre (str) : nom de l'oeuvre à lire et à analyser

        Returns :
            (void) : les ngrams et leurs fréquences sont entrés dans le dictionnaire passé en paramètre
        """

        fichier_oeuvre = open(oeuvre, "r", encoding="utf8")
        lignes = fichier_oeuvre.readlines()
        for ligne in lignes:
            # Traiter tous les mots de la ligne
            # Regrouper les mots en n-grammes (avec une fenêtre glissante)
            # Considérer que les espaces, tabulations et retours de chariot sont des séparateurs entre les mots
            # Mettre à jour dans le dictionnaire dict_de_ngrams avec le nombre d'occurrences des n-grammes trouvés
            # Les print qui suivent ne sont présents uniquement pour éliminer des avertissements
            # Les variables ligne et self devraient être utiles pour votre code
            print(ligne)
            print(self)
        fichier_oeuvre.close()
        return

    def analyze(self) -> None:
        """Fait l'analyse des textes fournis, en traitant chaque oeuvre de chaque auteur

        Args :
            (void) : toute l'information est contenue dans l'objet TextAn

        Returns :
            (void) : ne retourne rien, toute l'information extraite est conservée dans des structures internes
        """

        # Ajouter votre code ici pour traiter l'ensemble des oeuvres de l'ensemble des auteurs
        # Pour l'analyse :  faire le calcul des occurrences de n-grammes pour l'ensemble des oeuvres
        #   d'un certain auteur, sans distinction des oeuvres individuelles,
        #       et recommencer ce calcul pour chacun des auteurs
        #   En procédant ainsi, les oeuvres comprenant plus de mots auront un impact plus grand sur
        #   les statistiques globales d'un auteur.
        # Il serait possible de considérer que toutes les oeuvres d'un auteur ont un poids identique.
        #   Pour ce faire, il faudrait faire les calculs des occurrences pour chacune des oeuvres
        #       de façon indépendante, pour ensuite les normaliser,
        #       avant de les additionner pour obtenir le vecteur complet d'un auteur
        #   De cette façon, les mots d'un court poème auraient une importance beaucoup plus grande que
        #   les mots d'une très longue oeuvre du même auteur. Ce n'est PAS ce qui vous est demandé ici.
        #
        # Pour chaque auteur, créer un dictionnaire contenant :
        #       les ngrammes comme clé
        #       le nombre d'occurrences comme valeur
        #   Le dictionnaire de chacun des auteurs doit être ajouté à self.ngrams_auteurs, avec l'auteur comme clé

        # Ces trois lignes ne servent qu'à éliminer un avertissement. Il faut les retirer lorsque le code est complété
        ngram = self.get_empty_ngram(2)
        print("\t", ngram)
        print("\t", self.auteurs)

        # Le code qui suit indique comment accéder aux noms des fichiers qui contiennent les oeuvres des auteurs.
        # Vous pouvez l'adapter pour effectuer l'analyse
        # for auteur in self.auteurs:
        #     for oeuvre in self.auteurs[auteur]:
        #         print(oeuvre)
        # Pour chacun des auteurs, on devrait obtenir :
        #   self.mots_auteurs[auteur] = vecteur de n-grammes avec leur nombre d'occurrences
        #   self.normalized_mots_auteurs[auteur] = vecteur normalisé de n-grammes normalisés.

        for auteur in self.auteurs:
            oeuvres = self.get_aut_files(auteur)
            for oeuvre in oeuvres:
                print("\t", oeuvre)
        return

    def __init__(self) -> None:
        """Initialize l'objet de type TextAn lorsqu'il est créé

        Args :
            (void) : Utilise simplement les informations fournies dans la classe TextAnCommon

        Returns :
            (void) : Ne fait qu'initialiser l'objet de type TextAn
        """

        # Initialisation des champs nécessaires aux fonctions fournies
        super().__init__()

        # Au besoin, ajouter votre code d'initialisation de l'objet de type TextAn lors de sa création

        return
