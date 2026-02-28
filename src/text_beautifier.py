#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Code utilitaire améliorer le format d'un fichier de texte en créant des paragraphes,
    des lignes de longueur limitée, des phrases qui débutent par une majuscule.

    Copyright 2024-2026 F. Mailhot
"""
import re
import random


class TextBeautifier:
    """Classe utilisée pour rendre plus attrayant un texte généré automatiquement

    Copyright 2024-2026 F. Mailhot
    """

    def __init__(self) -> None:
        """Constructeur qui initialise la classe TextBeautifier

            - Ajoute une liste vide d'expressions régulières
            - Ajoute un ensemble d'expressions régulières à utiliser en séquence sur des chaînes de caractères à traiter

            Args :
                (void) : Tout le nécessaire se trouve dans le code d'initialisation de la classe TextBeautifier

            Returns :
                (void) : Au retour, l'objet est initialisé avec l'ensemble d'expressions régulières à appliquer

        """
        # Définit une liste d'expressions régulières pour modifier les textes générés
        #   Permet de produire un texte plus élégant...
        self.regex_list = []
        self.build_regex_list()

        return

    def add_swap_regex(self, match_str: str, swap_str_or_func: object) -> None:
        """Ajoute une expression régulière pour trouver une certaine séquence de caractères,
            ainsi qu'une expression régulière pour la remplacer

        Args :
            match_str (str) : Expression régulière pour trouver une chaîne de caractères
            swap_str_of_func (object) : Expression régulière pour remplacer la chaîne trouvée ou fonction à appliquer

        Returns :
           (void) : Ne fait qu'ajouter un tuple à la liste des expressions régulières de l'objet

        Copyright 2024-2026 F. Mailhot
        """
        self.regex_list.append((match_str, swap_str_or_func))
        return

    def build_regex_list(self) -> None:
        """Ajoute un ensemble d'expressions régulières pour modifier le texte généré et le rendre plus élégant

        Args :
            (void) : N'utilise que l'objet, où on ajoute l'ensemble d'expressions régulières

        Returns :
           (void) : Ne fait qu'ajouter une liste d'expressions régulières à l'objet

        Copyright 2024-2026 F. Mailhot
        """
        if len(self.regex_list) != 0:
            return
        # Mettre en majuscule la première lettre d'un mot qui suit la fin d'une phrase
        self.add_swap_regex(r"[\.!\?\-] [a-z]", self.convert_to_upper)

        # Mettre en majuscule la première lettre d'un mot qui suit un "_"
        self.add_swap_regex(r"\_[a-z]", self.convert_to_upper)

        # Mettre en majuscule la première lettre du premier mot du texte
        self.add_swap_regex("^.", self.convert_to_upper)

        # Traiter et préparer le texte
        self.add_swap_regex(r"\n", " ")
        self.add_swap_regex(r"  ", " ")
        self.add_swap_regex(r" \.", ".")
        self.add_swap_regex(r" \,", ",")
        self.add_swap_regex(r" \!", "!")
        self.add_swap_regex(r" ' ", "'")
        self.add_swap_regex(r" \;", ";")
        self.add_swap_regex(r" \:", ":")
        self.add_swap_regex(r" - ", "-")
        self.add_swap_regex(r"\( ", "(")
        self.add_swap_regex(r" \)", ")")
        self.add_swap_regex(r" m\.", " M.")
        self.add_swap_regex(r" mme", " Mme")
        self.add_swap_regex(r"_ (.*?) _", "_\\1_")
        self.add_swap_regex(r" _ ", " ")
        self.add_swap_regex(r"__", "")
        self.add_swap_regex(r"--", "")
        self.add_swap_regex(r"\.\)", ").")
        self.add_swap_regex(r"\._", "_.")
        return

    @staticmethod
    def replace(file_contents: str, match_str: str, swap_str_or_func) -> str:
        """Utilise une expression régulière pour identifier et modifier des éléments d'un texte

        Args :
            file_contents (str) : Texte complet à modifier
            match_str (str) : Expression régulière cible
            swap_str_or_func (str, callable) : Expression régulière pour modifier la cible (str) ou méthode à appliquer

        Returns :
           file_contents (str) : Texte modifié

        Copyright 2024-2026 F. Mailhot
        """
        file_contents = re.sub(match_str, swap_str_or_func, file_contents)
        return file_contents

    @staticmethod
    def convert_to_upper(match_object: re.Match) -> str :
        """Conversion de la chaîne de caractères en majuscule.  Typiquement utilisé sur un unique caractère

        Args :
            match_object (str) : Chaîne de caractères à convertir en majuscule

        Returns :
           (str) : Chaîne de caractères convertie en majuscule

        Copyright 2024-2026 F. Mailhot
        """
        return str(match_object.group().upper())

    @staticmethod
    def format_paragraphs(text: str, linemax: int = 100, pmax: int = 100, pvar: int = 40) -> str:
        """Modifie un texte comprenant une série de mots séparés par des espaces pour ajouter des paragraphes de
            tailles variables ainsi que des terminaisons de lignes pour limiter la longueur des lignes du texte

        Args :
            text (str) : Texte suivi (série de mots séparés par des espaces)
            linemax (int) : Nombre maximal de caractères sur une ligne
            pmax (int) : Nombre moyen de mots dans un paragraphe
            pvar (int) : Variation maximale du nombre de mots dans un paragraphe

        Returns :
           mod_text (str) : Texte modifié, avec des fins de paragraphes et de lignes

        Copyright 2024-2026 F. Mailhot
        """
        # 0. Extraire les mots du texte et ajouter un "." comme dernier mot (si nécessaire)
        words = text.split()
        if len(words) == 0:         # Si le texte à traiter est vide, retourner une chaîne de caractères vide
            return ""
        if words[-1][-1] != ".":    # Ajouter un "." à la toute fin du texte, s'il n'y en a pas déjà
            words.append(".")

        # 1. Créer des paragraphes (ajouter "\n\n" à la fin du dernier mot d'un paragraphe)
        # Taille des paragraphes variables, en moyenne pmax, avec une variation aléatoire de + ou - pvar mots
        # Les paragraphes se terminent avec un "mot" qui contient un '.' au dernier caractère
        paragraph_size = 0
        pmax_courant = pmax + pvar - random.random() * 2 * pvar
        for i in range(0, len(words)):
            paragraph_size += 1
            if (paragraph_size > pmax_courant) and (words[i][-1] == "."):
                words[i] += "\n\n"
                paragraph_size = 0
                pmax_courant = pmax + pvar - random.random() * 2 * pvar

        # 2. Ajouter un retour de chariot lorsque le dernier mot ferait dépasser la ligne de la taille linemax
        # Après la fin d'un paragraphe (présence de "\n\n" à la fin d'un mot), ajouter un alinéa ("\t")
        mod_text = "\t"
        line_size = 0
        for i in range(0, len(words)):
            word_size = len(words[i])
            if line_size + word_size + 1 > linemax:
                mod_text += "\n"
                line_size = 0
            mod_text += " " + words[i]
            if "\n\n" in words[i]:
                line_size = 0
                mod_text += "\t"
            else:
                line_size += word_size + 1
        return mod_text

    def post_traitement(self, file_contents: str) -> str:
        """Mettre des majuscules aux endroits appropriés, ajouter des fins de lignes et de paragraphes

        Args :
            file_contents (str) : Texte à traiter

        Returns :
           file_contents (str) : Le texte modifié

        Copyright 2024-2026 F. Mailhot
        """
        # Créer des paragraphes de taille agréable, séparer les lignes trop longues,
        #   ajouter un alinéa (tabulation, "\t") au début des paragraphes
        file_contents = self.format_paragraphs(file_contents)
        return file_contents

    def prettify_string(self, file_contents: str) -> str:
        """Modifie un fichier de texte pour le rendre plus attrayant :
            - Mise en forme des virgules et des points à la fin d'une phrase
            - Mise en forme des mots en début de phrase (majuscule pour la première lettre)
            - Ajout de séparation de paragraphes et de fin de lignes

        Args :
            textname (str) : Nom du fichier de texte à modifier

        Returns :
           (void) : Le fichier d'origine est remplacé par le texte modifié, alors rien n'est retourné

        Copyright 2024 F. Mailhot
        """
        # Appliquer les expressions régulières au texte :
        #   - Mettre en majuscules les mots en début de phrase
        for swap_tuple in self.regex_list:
            find_str = swap_tuple[0]
            swap_str = swap_tuple[1]
            file_contents = self.replace(file_contents, find_str, swap_str)

        # Création de paragraphes de taille aléatoire mais raisonnable
        new_file_contents = self.post_traitement(file_contents)

        return new_file_contents

    def prettify_file(self, textname: str) -> None:
        """Modifie un fichier de texte pour le rendre plus attrayant :
            - Mise en forme des virgules et des points à la fin d'une phrase
            - Mise en forme des mots en début de phrase (majuscule pour la première lettre)
            - Ajout de séparation de paragraphes et de fin de lignes

        Args :
            textname (str) : Nom du fichier de texte à modifier

        Returns :
           (void) : Le fichier d'origine est remplacé par le texte modifié, alors rien n'est retourné

        Copyright 2024-2026 F. Mailhot
        """
        # https://www.studytonight.com/python-howtos/search-and-replace-a-text-in-a-file-in-python
        with open(textname, "r+", encoding='utf8') as file:
            # Lire le contenu du fichier, pour le modifier en mémoire
            file_contents = file.read()

            # Modifier le contenu du fichier pour que le texte soit plus joli
            pretty_file_contents = self.prettify_string(file_contents)

            # Réécriture du fichier de départ avec le texte modifié
            file.seek(0)
            file.truncate()
            file.write(pretty_file_contents)

        return
