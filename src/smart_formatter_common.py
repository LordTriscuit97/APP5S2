#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utilitaire pour imprimer les commentaires d'aide sur plusieurs lignes (dans configargparse)

    Note :
        - La classe SmartFormatter permet d'imprimer correctement
            la chaîne de caractère "help" lorsqu'elle débute par "R|"

    Copyright 2024-2026 Frédéric Mailhot
"""
import configargparse


class SmartFormatter(configargparse.HelpFormatter):
    """Classe destinée à formatter l'affichage des messages d'aide de configargparse

        - Méthode incluse : _split_lines, qui remplace ce qui est fourni par défaut par configargparse

    """

    def _split_lines(self, text: str, width: int) -> list[str]:
        if text.startswith('R|'):
            return text[2:].splitlines()
            # this is the RawTextHelpFormatter._split_lines
        return configargparse.HelpFormatter._split_lines(self, text, width)
