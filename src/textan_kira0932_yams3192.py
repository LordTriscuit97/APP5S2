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
import math
from collections import Counter
import random

from sympy import true
from sympy.physics.units import frequency

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
    PONC = ["!", ";", ",", ".", "?", ":", "-","«", "»"]

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

        size = 0
        for valeur in dict_de_ngrams.values():
            size += valeur * valeur

        size = math.sqrt(size)
        return size

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
        norm_dict = {}
        somme_carres = sum(valeur ** 2 for valeur in dict_de_ngrams.values())
        norme_l2 = math.sqrt(somme_carres)

        if norme_l2 == 0:
            return {}

        # Diviser chaque élément par la norme
        for cle, valeur in dict_de_ngrams.items():
            norm_dict[cle] = valeur / norme_l2

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
        #addition des dict avec counter
        sum_dict = Counter(dict1)+Counter(dict2)

        #passage au dict
        return dict(sum_dict)

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

        scal_prod = 0
        for ngram, compte in dict1.items():
            if ngram in dict2:
                scal_prod += compte * dict2[ngram]
        return scal_prod

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

        #extraction des ngrammes
        dict_oeuvre = {}
        self.compute_ngram_stats(dict_oeuvre, oeuvre)

        #angle entre l'oeuvre et celles des auteurs
        l_resultats = []
        dict_oeuvre = self.normalize_vector(dict_oeuvre)
        for auteur in self.auteurs:
            angle = self.dot_product_dict(dict_oeuvre, self.normalized_ngrams_auteurs[auteur])
            l_resultats.append((auteur, angle))


        return l_resultats

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

    def get_ngram_occurrence(self, auteur: str, ngram) -> int:
        """Retourne le nombre d'occurrences du n-gramme pour cet auteur

        Args :
            auteur (string) : le nom de l'auteur

            ngram (objet du type utilisé pour un ngram) : le n-gramme dont on désire la fréquence

        Returns :
            (int) : retourne le nombre d'occurrences du n-gramme pour l'auteur donné

        Copyright 2024-2026 F. Mailhot
        """
        #aller chercher le dict de l'auteur
        dict_auteur = self.ngrams_auteurs.get(auteur, {})

        #trouver le nombre d'occurences
        occurence = dict_auteur.get(ngram,0)

        return occurence

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

        # aller chercher le dict de l'auteur
        dict_auteur = self.ngrams_auteurs.get(auteur, {})

        total_occ = 0
        for valeur in dict_auteur.values():
            total_occ += valeur

        return total_occ

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
        #optimisation (triage)
        options_suivantes = {}
        taille_contexte = 1

        for ngram, freq in auteur_dict.items():
            if len(ngram) > 1:
                taille_contexte = len(ngram) - 1
                contexte = ngram[:-1] #tous les mots du ngram sauf le dernier
                mot_cible = ngram[-1] #le dernier mot

                #création liste si le mot est nouveau
                if contexte not in options_suivantes:
                    options_suivantes[contexte] = []

                options_suivantes[contexte].append((mot_cible, freq))


        #dictionnaire vide
        if not options_suivantes:
            return

        #génératino premier mot
        l_ngrams = list(auteur_dict.keys())
        l_occurences = list(auteur_dict.values())

        #mot random
        mot_choisi = random.choices(l_ngrams, weights=l_occurences, k=1)[0]

        #tuple->liste
        l_mots_gen = list(mot_choisi)

        #chaine de markov
        while len(l_mots_gen) < taille:

            contexte_actuel = tuple(l_mots_gen[-taille_contexte:])

            choix_possibles = options_suivantes.get(contexte_actuel, [])

            #gestion quand un mot n'a pas de mot suivant (choisir nouveau)
            if not choix_possibles:
                nouveau_depart = random.choices(l_ngrams, weights=l_occurences, k=1)[0]
                l_mots_gen.extend(list(nouveau_depart))
                continue

            #sélection nouveau mot
            mots_candidats = [choix[0] for choix in choix_possibles]
            poids = [choix[1] for choix in choix_possibles]

            nouveau_mot = random.choices(mots_candidats, weights=poids, k=1)[0]
            l_mots_gen.append(nouveau_mot)

        #écriture fichier
        phrase_finale = " ".join(l_mots_gen)
        print(phrase_finale, file=to_file)

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
        if auteur not in self.ngrams_auteurs:
            return []
        dict_auteur = self.ngrams_auteurs[auteur]

        # Transformer en liste de tuples
        liste_tuples = [(freq, gram) for freq, gram in dict_auteur.items()]

        # On tri la liste dans l'ordre décroissant
        liste_tuples.sort(key=lambda x: x[0], reverse=True)

        if not liste_tuples or k > len(liste_tuples) or k <= 0:
            return []

        ngram_choisi = list(liste_tuples[k-1][1])
        return ngram_choisi


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
        # Liste temporaire pour stocker tous les mots de l'oeuvre après nettoyage
        tous_les_mots =  []

        fichier_oeuvre = open(oeuvre, "r", encoding="utf8")
        lignes = fichier_oeuvre.readlines()
        for ligne in lignes:
            # Traiter tous les mots de la ligne
            # Regrouper les mots en n-grammes (avec une fenêtre glissante)
            # Considérer que les espaces, tabulations et retours de chariot sont des séparateurs entre les mots
            # Mettre à jour dans le dictionnaire dict_de_ngrams avec le nombre d'occurrences des n-grammes trouvés
            # Les print qui suivent ne sont présents uniquement pour éliminer des avertissements
            # Les variables ligne et self devraient être utiles pour votre code

            # On nettoie la ligne actuelle
            mots_ligne = self.nettoyerTexte(ligne)
            # On ajoute les mots de cette ligne à notre liste globale
            tous_les_mots.extend(mots_ligne)
        fichier_oeuvre.close()

        # Application de la fentre glissante
        n = self.ngram_size
        for i in range (len(tous_les_mots) - n + 1):
            # Création du n-gramme (sous forme de tuple pour servir de clé)
            ngram = tuple(tous_les_mots[i: i + n])

            # Mise à jour du dictionnaire passé en paramètre
            if ngram in dict_de_ngrams:
                dict_de_ngrams[ngram] += 1
            else:
                dict_de_ngrams[ngram] = 1
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

        # Le code qui suit indique comment accéder aux noms des fichiers qui contiennent les oeuvres des auteurs.
        # Vous pouvez l'adapter pour effectuer l'analyse
        # for auteur in self.auteurs:
        #     for oeuvre in self.auteurs[auteur]:
        #         print(oeuvre)
        # Pour chacun des auteurs, on devrait obtenir :
        #   self.mots_auteurs[auteur] = vecteur de n-grammes avec leur nombre d'occurrences
        #   self.normalized_mots_auteurs[auteur] = vecteur normalisé de n-grammes normalisés.
        for auteur in self.auteurs:
            #On initialise un dictionnaire vide pour cet auteur
            stats_auteur = {}

            # Récupérer les fichiers de cet auteur
            oeuvres = self.get_aut_files(auteur)

            for oeuvre in oeuvres:
                #ngram
                self.compute_ngram_stats(stats_auteur, oeuvre)

                #stat auteur
                self.ngrams_auteurs[auteur] = stats_auteur.copy()

                #vecteur unitaire
                self.normalized_ngrams_auteurs[auteur] = self.normalize_vector(stats_auteur)

    def nettoyerTexte(self, texte) -> list:
        #normaliser le string
        texte = self.normalize_string(texte)

        #majuscules en minuscules
        texte = texte.lower()

        #gestion ponctuation
        for punc in self.PONC:
            texte = texte.replace(punc, " " + punc + " ")

        #enlever les espaces
        texte = texte.split()

        return texte



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
        # 1. Initialise le dictionnaire des ngrams par auteur
        self.ngrams_auteurs = {}

        # 2. Initialise le dictionnaire des ngrams normalisés par auteur
        self.normalized_ngrams_auteurs = {}
        return
