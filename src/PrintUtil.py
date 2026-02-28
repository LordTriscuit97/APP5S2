#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Code utilitaire pour gérer l'impression

    Copyright 2025-2026 F. Mailhot
"""
import io
import sys
import os
from typing import TextIO


class PrintUtil:
    """Classe permettant la gestion de l'impression :

        - Permet de définir où se fera l'impression des messages de test (écran ou fichier de log)
        - Permet de désactiver les impressions effectuées dans le code étudiant
        - Cette classe est utilisée directement, sans instance, en utilisant uniquement :
            - des attributs de classe
            - des méthodes de classe
        - Les attributs de classe utilisés sont :
            - __STDOUT : sys.stdout d'origine : ne peut être modifié, conserve le "vrai" sys.stdout
                - utilisé pour revenir à sys.stdout après blocage des print()
            - _current : sys.stdout par défaut, peut être redéfini, utilisé par log_print()

    Copyright 2025-2026 F. Mailhot
    """
    # Voir: https://www.toptal.com/python/python-class-attributes-an-overly-thorough-guide

    # __STDOUT : L'intention ici est de déclarer une constante, initialisée au départ et jamais modifiée par la suite
    __STDOUT = sys.stdout

    # _current: Là où l'impression se fera.  On peut modifier cette variable
    _current = sys.stdout

    @classmethod
    def set_stdout_path(cls, path: str) -> None:
        """Redéfinit le fichier courant pour l'impression effectuée par la classe (méthode log_print) :
            - Permet d'identifier où se fera l'impression (sys.stdout habituellement, ou fichier de log)

        Args :
            path (str) : Le chemin complet vers le nouveau fichier de log

        Returns :
            (void) : Au retour, le fichier d'impression des logs est redéfini pour la méthode log_print

        Copyright 2025-2026 F. Mailhot
        """
        cls.close_stdout()  # Fermer le fichier de log s'il était utilisé
        cls._current = open(path, "w", encoding="utf-8")
        return

    @classmethod
    def reset_stdout(cls) -> None:
        """Remet sys.stdout comme fichier d'impression :

        Returns :
            (void) : Au retour, le fichier courant d'impression a été fermé et remplacé par sys.stdout

        Copyright 2025-2026 F. Mailhot
        """
        cls.close_stdout()
        cls._current = cls.__STDOUT
        return

    @classmethod
    def get_stdout(cls) -> TextIO:
        """Retourne le pointeur vers le fichier utilisé pour l'impression avec la méthode log_print :

        Returns :
            (TextIO) : Au retour, le fichier courant (pour l'impression) est retourné

        Copyright 2025-2026 F. Mailhot
        """
        return cls._current

    @classmethod
    def close_stdout(cls) -> None:
        """Ferme le fichier associé à l'impression par la méthode log_print :
            - Appelé à la fin, pour fermer correctement le fichier de log
            - Si le "fichier" de log est l'écran (sys.stdout), ne fait rien

        Copyright 2025-2026 F. Mailhot
        """
        if cls._current != cls.__STDOUT:
            cls._current.close()
            cls._current = cls.__STDOUT
        return

    @classmethod
    def get_sys_stdout(cls) -> TextIO:
        """Retourne la version originale de sys.stdout :

       Returns :
            (TextIO) : Au retour, sys.stdout original est retourné

        Copyright 2025-2026 F. Mailhot
        """
        return cls.__STDOUT

    @classmethod
    def block_stdout(cls) -> None:
        """Redéfinit sys.stdout pour l'impression dans le code étudiant (print standards) :
            - Remplace sys.stdout par /dev/null (impression inactive)

        Returns :
            (void) : Au retour, sys.stdout est redéfini vers os.devnull

        Copyright 2025-2026 F. Mailhot
        """
        sys.stdout = open(os.devnull, "w")
        return

    @classmethod
    def unblock_stdout(cls) -> None:
        """Remet la valeur par défaut de sys.stdout :
            - permet d'imprimer de nouveau normalement avec print()
            - n'est utilisé que si block_stdout a été utilisé auparavant

        Returns :
            (void) : Au retour, sys.stdout original est revenu

        Copyright 2025-2026 F. Mailhot
        """
        if sys.stdout != cls.__STDOUT:
            sys.stdout.close()
            sys.stdout = cls.__STDOUT
        return

    @classmethod
    def log_print(cls, *args, **kwargs) -> None:
        """Imprime dans le fichier de log :
            - Par défaut, si ce qui est imprimé se termine par un retour de chariot/newline
                (end='newline/carriage return' par défaut), alors effectuer un flush après impression
            - Pour contrôler précisément cette action, "flush=True/False" peut être utilisé comme paramètre

        Args :
            args :      Paramètres habituels de print()
            kwargs :    Paramètres habituels de print()

        Returns :
            (void) : Au retour, l'impression a eu lieu dans le fichier de log (par défaut, sys.stdout)

        Copyright 2025-2026 F. Mailhot
        """

        # Imprime dans le fichier de log (flush seulement si fin de ligne)
        end = kwargs.get("end", "\n")

        # Permet à l'appelant de forcer/empêcher le flush : PrintUtil.log_print(..., flush=True/False)
        flush_override = kwargs.pop("flush", None)

        print(*args, **kwargs, file=cls._current)

        if flush_override is True:
            cls._current.flush()
        elif flush_override is False:
            return
        else:
            # Auto : flush seulement si ce qui vient d'être imprimé termine une ligne
            if isinstance(end, str) and end.endswith("\n"):
                cls._current.flush()
        return
