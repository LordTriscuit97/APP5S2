#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utilitaires pour traiter les chaînes de caractères unicode (utf-8), nécessaires pour les lettres accentuées

    Copyright 2024-2026 Frédéric Mailhot
"""
from unicodedata import normalize
import io


class HandleUnicodeCommon:
    """Classe destinée à traiter uniformément les caractères utf-8

    - Méthode utile pour le laboratoire et la problématique :
        - normalize_string : retourne une chaîne de caractères utf-8 normalisés

    - Méthodes utiles pour le laboratoire seulement :
        - debug_utf8_string : imprime sous forme hexadecimale une chaîne de caractères utf-8
        - string_from_hex_list : transforme une liste de caractères hexadécimaux en chaîne de caractères utf-8
        - get_strings : produit deux chaînes de caractères utf-8 en apparence identiques (utile pour le laboratoire)

"""

    @staticmethod
    def normalize_string(a_str: str) -> str:
        """Retourne une chaîne de caractères normalisée.
            Cette opération est nécessaire pour les lettres accentuées, représentées en format UTF-8.
            En effet, les lettres accentuées ont deux représentations possibles :
            - la lettre accentuée elle-même, avec un code unique (toutes les lettres accentuées existent en UTF-8)
            - la lettre non-accentuée, suivie d'un caractère qui indique le type d'accent
            Le problème est que Python ne gère qu'en partie les deux représentations possibles :
            - à l'impression de la chaîne de caractères, il n'y aura pas de différence
            - la comparaison indiquera cependant qu'il ne s'agit pas de la même chaîne de caractères.
            Il est donc essentiel d'avoir une représentation unique (ce que la méthode normalize_string() effectue).

        Args :
            a_str (str) : La chaîne de caractères à convertir

        Returns :
            (str) : La chaîne de caractères avec une représentation UTF-8 canonique (unique et standard)

        Copyright 2024-2026 F. Mailhot
        """

        def NFC(s: str) -> str:
            return normalize('NFC', s)  # Voir https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize

        # Note : il faut normaliser deux fois pour assurer une représentation canonique (d'où NFC(NFC(...)))
        # Voir https://docs.python.org/3/howto/unicode.html
        return NFC(NFC(a_str))

    @staticmethod
    def debug_utf8_string(a_str: str) -> str:
        """Imprime (dans une chaine de caractères utf-8) sous forme hexadécimale
           tous les caractères utf-8 d'une chaîne de caractères passée en paramètre.
           Utile uniquement pour le débogage de chaînes de caractères avec des lettres accentuées.
           Il y a plusieurs formes possibles d'une chaîne de caractères utf-8.
           Python ne les distingue pas à l'impression, mais peut tout de même indiquer
           que les chaînes de caractères sont différentes.

           Cette méthode permet de vérifier le contenu réel d'une chaîne de caractères,
           en affichant TOUS les caractères.

        Args :
            a_str (str) : La chaîne de caractères à observer

        Returns :
            str_res (str) : Cette méthode (utilisée pour déboguer) retourne dans une chaîne de caractères
            en format utf-8 le contenu de la chaîne de caractères sous forme hexadécimale passée en paramètre

        Copyright 2024-2026 F. Mailhot
        """
        # Inspiré de https://stackoverflow.com/questions/39823303/python3-print-to-string
        with io.StringIO() as output_str:
            print("Contenu unicode (utf-8) de ", a_str, ": ", file=output_str, end="")
            for character in a_str:
                print("(", character, " : ", character.encode('utf-8').hex(), ") ", sep="", file=output_str, end="")
            str_res = output_str.getvalue()
            output_str.close()
        return str_res

    @staticmethod
    def string_from_hex_list(hex_utf8_str: str) -> str:
        """Transforme un ensemble de nombres hexadécimaux (sous forme de chaîne de caractères)
            en chaîne de caractères utf-8

        Args :
            hex_utf8_str (str) : Chaîne de caractères en format hexadecimal à convertir en chaîne de caractères utf8

        Returns :
            (str) : La chaine de caractères utf-8 résultante

        Copyright 2024-2026 F. Mailhot
        """
        # Adapté de https://sparkbyexamples.com/python/python-hex-to-string/
        # et de https://discuss.python.org/t/how-to-convert-character-to-hexadecimal-and-back/25056/6

        bytes_obj = bytes.fromhex(hex_utf8_str)
        utf8_string = bytes_obj.decode('utf-8')
        return utf8_string

    @staticmethod
    def get_strings() -> tuple[str, str]:
        hex_list1 = ""
        hex_list2 = ""
        list1 = ['c3a9', '74', '6f', '6e', '6e', '61', '6e', '74']
        for hex_code in list1:
            hex_list1 += hex_code
        word1 = HandleUnicodeCommon.string_from_hex_list(hex_list1)
        list2 = ['65', 'cc81', '74', '6f', '6e', '6e', '61', '6e', '74']
        for hex_code in list2:
            hex_list2 += hex_code
        word2 = HandleUnicodeCommon.string_from_hex_list(hex_list2)
        return word1, word2
