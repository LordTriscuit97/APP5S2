#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Code utilitaire pour implémenter le patron de conception "command", utilisé dans test_textan.py

    Copyright 2024-2026 F. Mailhot
"""

import debug_handler_common
import re
from PrintUtil import PrintUtil


class ExecOperation:
    """Classe qui définit les opérations à effectuer par la classe CommandTextan :

        - Permet de définir les méthodes à exécuter et les commentaires à imprimer avant et/ou après l'exécution

    Copyright 2024-2026 F. Mailhot
    """

    DEFAULT_POST_PRINT = "\tExécution"
    REPLACE_CIP = "X_CIP_X"

    def __init__(self,
                 exec_flag: bool,
                 exec_pre_print: str,
                 exec_func: callable,
                 func_name: str,
                 exec_has_param: bool,
                 exec_post_print_flag: bool,
                 exec_post_print: str) -> None:

        """Constructeur pour la classe ExecOperation : Prépare l'exécution d'une commande

        Args :
            exec_flag (bool) : Indique que cette méthode doit être exécutée
            exec_pre_print (str) : Message à imprimer avant l'exécution
            exec_func (callable) : Méthode à exécuter
            exec_has_param (bool) : La méthode utilise un paramètre (le cip) qui est passé en paramètre
            exec_post_print_flag(bool) : Imprime (ou non) exec_post_print
            exec_post_print (str) : Message à imprimer après l'exécution

        Returns :
            (void) : Au retour, la nouvelle instance d'exécution de commande est prête à être utilisée
        """

        self.exec_flag = exec_flag
        self.exec_pre_print = exec_pre_print
        self.exec_func = exec_func
        self.func_name = func_name
        self.exec_has_param = exec_has_param
        self.exec_post_print_flag = exec_post_print_flag
        if exec_post_print == "":
            exec_post_print = ExecOperation.DEFAULT_POST_PRINT
        self.exec_post_print = exec_post_print
        return

    def run_op(self, cip: str) -> bool:
        """Exécution d'une commande prédéfinie

        Args :
            cip (str) : Le cip à utiliser en paramètre pour les méthodes qui le requièrent

        Returns :
            (void) : Au retour, la méthode a été exécutée, avec les messages imprimés avant et après
        """
        if self.exec_flag:
            exec_pre_print = re.sub(ExecOperation.REPLACE_CIP, cip, self.exec_pre_print)
            exec_post_print = re.sub(ExecOperation.REPLACE_CIP, cip, self.exec_post_print)
            PrintUtil.log_print(exec_pre_print)

            if self.exec_has_param:
                self.exec_func(cip)
            else:
                self.exec_func()

            if self.exec_post_print_flag:
                PrintUtil.log_print(exec_post_print, end="")
        return self.exec_flag


class CommandTextan:
    """Classe qui implémente le patron de conception "command" :

        - Permet de prédéfinir l'ordre des méthodes à appeler, puis d'en faire l'exécution

    Copyright 2024-2026 F. Mailhot
    """
    def exec_operations(self, cip: str) -> None:
        """Appelle dans l'ordre la série de méthodes enregistrées dans le champ self.operations

        Args :
            cip (string) : Le ou les CIPs des membres de l'équipe dont le code est exécuté

        Returns :
            (void) : Rien n'est retourné : au retour, l'ensemble des méthodes a été exécuté
        """
        for operation in self.operations:
            self.debug_handler.set_start_time2()  # Démarre le chronomètre pour mesurer le temps d'exécution
            did_something = operation.run_op(cip)
            if did_something and operation.exec_post_print_flag:
                PrintUtil.log_print(f" en {self.debug_handler.stop_execution_timing2():.2f} secondes")
        return

    def register_one_operation(self,
                               exec_flag: bool,
                               exec_pre_print: str,
                               exec_func: callable,
                               func_name: str,
                               exec_has_param: bool,
                               exec_post_print_flag: bool,
                               exec_post_print: str) -> None:
        """Enregistre une méthode à exécuter pour vérifier le code.
            Les enregistrements doivent être faits dans l'ordre dans lequel leur exécution doit s'effectuer

        Args :
            exec_flag (bool) : Indique si cette méthode doit être exécutée ou non
            exec_pre_print (string) : Message à imprimer avant l'exécution
            exec_func (callable) : Méthode à exécuter
            func_name (string) : Nom de la fonction à appeler et courte explication
            exec_has_param (bool) : Indique si la méthode exec_func doit utiliser un paramètre ou non (le cip)
            exec_post_print (string) : Message à imprimer après l'exécution de la méthode

        Returns :
            (void) : Rien n'est retourné : au retour, la méthode a été enregistrée
        """
        new_op = ExecOperation(exec_flag,
                               exec_pre_print,
                               exec_func,
                               func_name,
                               exec_has_param,
                               exec_post_print_flag,
                               exec_post_print)
        self.operations.append(new_op)
        return

    def __init__(self) -> None:
        """Constructeur pour la classe CommandTextan.  Initialisation de l'ensemble des éléments requis

        Args :
            (void) : Le constructeur initialise l'objet CommandTextan

        Returns :
            (void) : Au retour, la nouvelle instance est prête à être utilisée
        """

        super().__init__()
        self.operations = []
        self.debug_handler = debug_handler_common.DebugHandler()

        return
