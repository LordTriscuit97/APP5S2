#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Code utilitaire pour lire et interpréter la ligne de commande pour test_textan.py

    Copyright 2018-2026 F. Mailhot
"""

# Info sur configargparse : https://github.com/bw2/ConfigArgParse :
#   Supporte de façon unifiée des fichiers de configuration et des paramètres en ligne de commande
import configargparse
import sys
import os
import timeit
from handle_unicode_common import HandleUnicodeCommon
import debug_handler_common
from PrintUtil import PrintUtil


class ParsingClassTextAn:
    """Classe qui effectue la lecture des paramètres d'exécution :
        - Permet de lire le fichier de configuration (test_textan_config.yml)
        - Permet de lire les paramètres de la ligne de commande
        - Les paramètres de la ligne de commande ont priorité sur ceux du fichier de configuration

    Copyright 2024-2026 F. Mailhot
    """
    def parse_cli(self) -> None:
        """Utilise le module configargparse pour :
            - Enregistrer les commandes à reconnaître
            - Lire le fichier de configuration
            - Lire la ligne de commande
            - Créer le champ self.args qui récupère la structure produite par la lecture des paramètres
        Args :
            (void) : Rien n'est passé en paramètre, tout est extrait par configargparse

        Returns :
            (void) : Au retour, toutes les commandes reconnues sont comprises dans self.args
        """
        config_file_path = os.path.join(os.path.curdir, "test_textan_config.yml")
        parser = configargparse.ArgParser(
            default_config_files=[config_file_path],
            prog="test_textan.py"
        )

        # Définit un fichier de configuration différent de celui par défaut
        parser.add_argument(
            '-config', '--config',
            is_config_file=True,
            help='Nom du fichier de configuration différent de celui par défaut',
        )
        # Répertoire où se trouvent les auteurs, leurs oeuvres, la liste d'étudiants, la liste d'oeuvres inconnues
        parser.add_argument(
            "--d",
            help="Répertoire contenant les sous-repertoires des auteurs \
                            (TextesPourEtudiants par défaut)",
        )
        # Indication du répertoire où se trouvent les programmes étudiants, par rapport au répertoire de travail
        parser.add_argument(
            "--dir_code",
            type=str,
            help="Répertoire contenant la liste des CIPs et leur code textan_CIP1_CIP2.py ('.' par défaut)",
        )
        # Fichier qui contient la liste des cips pour lesquels on doit exécuter le code
        parser.add_argument(
            "--e",
            help="Nom du fichier qui contient la liste des CIP1_CIP2 \
                            pour exécuter textan_CIP1_CIP2.py \
                            (etudiants.txt par défaut)",
        )

        # Type de n-gramme à utiliser pour l'analyse et la génération aléatoire de textes
        parser.add_argument(
            "--m",
            type=int,
            choices=range(1, 21),
            help="Mode (1 ou 2 ou 3 ou ... 20) - (4 par défaut) \
                            unigrammes ou digrammes ou trigrammes ou ... 20-grammes",
        )

        # Indication du K-ième plus fréquent n-gramme à imprimer pour chacun des auteurs
        parser.add_argument(
            "--K",
            type=int,
            help="Indication du K-ième plus fréquent n-gramme a imprimer pour chacun des auteurs",
        )

        # Répertoire où trouver les fichiers inconnus et leur liste (-F)
        parser.add_argument(
            "--f_dir",
            help="Nom du répertoire où se trouvent les fichiers inconnus et leur liste",
        )
        # Fichier unique pour lequel on recherche l'auteur
        parser.add_argument(
            "--f",
            help="Fichier inconnu pour lequel on recherche un auteur",
        )
        # Liste de fichiers pour lesquels on recherche l'auteur
        parser.add_argument(
            "--F",
            help="Nom du fichier qui contient la liste des fichiers inconnus pour lesquels on recherche un auteur",
        )

        # Les arguments qui suivent sont tous liés à la génération de texte aléatoire
        # Génération d'un texte aléatoire pour cet auteur
        parser.add_argument(
            "--g",
            help="Génération de texte aléatoire pour cet auteur spécifique",
        )
        # Génération d'un texte aléatoire basé sur les fréquences normalisées de tous les auteurs identifiés
        parser.add_argument(
            "--G",
            help="Génération d'un texte aléatoire par auteur identifié dans le fichier indiqué",
        )
        # Génération d'un texte aléatoire par auteur identifié
        parser.add_argument(
            "--G_fusion",
            help="Génération de texte avec les statistiques combinées des auteurs identifiés dans le fichier indiqué",
        )
        parser.add_argument(
            "--g_size",
            type=int,
            help="Taille du texte à générer (500 mots par défaut)",
        )
        parser.add_argument(
            "--g_name",
            type=str,
            help="R|"
                 "Nom du fichier à générer, avec jetons:"
                 "<CIP>: cip de l'équipe"
                 "<AUT>: nom de l'auteur"
                 "<DATE>: date de calendrier"
                 "<HR>: heure de création"
                 "<MIN>: minute"
                 "<SEC>: seconde"
        )
        parser.add_argument(
            "--g_dir",
            type=str,
            help="Répertoire où les fichiers à générer seront créés"
        )
        # Par défaut, les textes générés sont passés par le "text prettifier".
        # Ce paramètre permet de ne pas le faire, s'il est mis à False
        parser.add_argument(
            "--pretty",
            type=bool,
            help="Utiliser (ou non) l'appel au système de reformattage des textes générés pour les rendre plus beaux",
        )

        # Mode verbose (imprime beaucoup d'information de débogage)
        parser.add_argument("--v", action="store_true", help="Mode verbose")

        # Retire complètement les signes de ponctuation
        parser.add_argument(
            "--noPonc", action="store_true", help="Retirer la ponctuation"
        )

        # Nombre maximal d'appels récursifs (par défaut, Python en permet 1000)
        parser.add_argument(
            "--recursion", help="Récursion maximale permise (Python en permet 1000 par défaut)"
        )

        parser.add_argument(
            "--golden",
            type=str,
            help="Compare les résultats avec la version 'golden' indiquée par ce paramètre",
        )

        # Tous les prints se feront dans ce fichier (fichier de log).
        parser.add_argument(
            "--fichier_res",
            type=str,
            help="Tous les prints seront faits dans ce fichier"
        )
        # Tous les fichiers créés le seront dans ce répertoire
        parser.add_argument(
            "--dir_res",
            type=str,
            help="Tous les résultats seront ajoutés dans ce répertoire (sous le répertoire courant)",
        )
        # Indique que les print apparaissant dans le code étudiant ne seront pas actifs
        parser.add_argument(
            "--no_stdout_etudiant",
            action="store_true",
            help="Empêcher l'impression par le code étudiant",
        )

        # Limite le temps maximal d'exécution (en secondes) de chaque programme étudiant
        parser.add_argument(
            "--timeout",
            type=int,
            help="Temps maximum (secondes) pour l'exécution du système (120 secondes par défaut)",
        )

        # Comparaison des auteurs (calcule le cosinus de l'angle entre les vecteurs des différents auteurs)
        parser.add_argument(
            "--compare_auteurs",
            action="store_true",
            help="Indique les proximités des textes des différents auteurs, en calculant le cosinus de l'angle",
        )

        self.parser = parser
        self.args = parser.parse_args()
        return

    @staticmethod
    def read_lines(fichier_a_lire: str) -> list[str]:
        """Lit les lignes d'un fichier et retourne un tableau contenant une entrée par ligne :

        Args :
            fichier_a_lire (str) : Nom d'un fichier contenant une chaîne de caractères par ligne

        Returns :
            list : retourne une liste avec une entrée par ligne (une chaîne de caractères par ligne)
        """
        lignes = []
        f = open(fichier_a_lire, "r", encoding='utf8')
        for line in f:
            lignes.append(HandleUnicodeCommon.normalize_string(line.strip()))
        return lignes

    def set_log_file(self) -> None:
        """Prépare l'impression (à l'écran ou dans un fichier de log) :
            - Utilise la classe PrintUtil pour déterminer le type d'impression
            - Désactive au besoin l'impression dans le code étudiant

        Args :
            (void) : Rien n'est passé en paramètre, tout vient de l'objet

        Returns :
            (void) : L'impression est prête à être effectuée
        """

        if self.no_stdout_print_etudiant:  # Désactivation de l'impression standard par le code étudiant
            PrintUtil.block_stdout()

        if not self.fichier_res:  # Pas de setup de logs si le fichier de résultat n'a pas été demandé
            return

        # https://stackoverflow.com/questions/5104957/how-do-i-create-a-file-at-a-specific-path
        cur_path = os.path.dirname(__file__)
        if self.dir_res:
            dir_res_path = os.path.join(str(cur_path), str(os.path.relpath(self.dir_res, cur_path)))
            try:  # https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
                os.mkdir(dir_res_path)
            except FileExistsError:
                pass
        else:
            dir_res_path = cur_path

        # Préparer le chemin (path) et ouvrir le fichier de résultats
        #   Voir: https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
        #   et: https://stackoverflow.com/questions/3597480/how-to-make-python-3-print-utf8
        output_file_path = os.path.join(dir_res_path, self.fichier_res)
        PrintUtil.set_stdout_path(output_file_path)
        self.dir_res_path = dir_res_path
        return

    def prepare_dir_code(self):
        """Ajoute le répertoire de code au "path" :

        Args :
            (void) : Toute l'information requise est présente dans l'objet

        Returns :
            (void) : Au retour, le répertoire de code est accessible
        """
        if self.dir_code != ".":
            # Ajout de ce répertoire dans sys.path pour trouver les fichiers textan_cip.py à charger
            # https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
            sys.path.append(self.dir_code)
        return

    def prepare_get_kth_ngram(self):
        """Indique si le calcul du k-ième n-gramme doit être effectué :

        Args :
            (void) : Prépare la détermination du k-ième n-gramme, si approprié

        Returns :
            (void) : Au retour, l'attribut est configuré
        """
        if self.kth_ngram != 0:
            self.do_get_kth_ngram = True
        return

    def prepare_single_author_generation(self):
        """Indique si la génération de texte (un seul auteur) doit être effectuée :

        Args :
            (void) : L'information requise est présente dans l'objet

        Returns :
            (void) : Au retour, l'attribut est configuré
        """
        if self.auteur_unique_pour_generation:
            self.do_gen_text = True
            self.auteurs.append(
                HandleUnicodeCommon.normalize_string(self.auteur_unique_pour_generation))
        return

    def prepare_multiple_author_generation(self):
        """Indique si la génération de textes multiples (un par auteur identifié) doit être effectuée :

        Returns :
            (void) : Au retour, la liste d'auteurs est configurée
        """
        if self.liste_auteurs_pour_generation:
            self.do_gen_text = True
            liste_auteurs = os.path.join(self.g_dir,
                                         self.liste_auteurs_pour_generation)
            self.auteurs = self.read_lines(liste_auteurs)
        return

    def prepare_fused_authors_generation(self):
        """Indique si la génération de textes d'auteurs fusionnés doit être effectuée :
            - Les statistiques de l'ensemble des auteurs identifiés seront combinées

        Returns :
            (void) : Au retour, la liste d'auteurs est configurée, les attributs sont configurés
        """
        if self.liste_auteurs_pour_generation_commune:
            self.do_gen_text = True
            self.do_fused_gen_text = True
            liste_auteurs = os.path.join(self.g_dir,
                                         self.liste_auteurs_pour_generation_commune)
            self.auteurs = self.read_lines(liste_auteurs)
        return

    def prepare_generation(self):
        """Prépare la génération de textes aléatoires :
            - Configuration pour génération à partir des statistiques d'un auteur unique
            - Ou bien génération de textes multiples pour tous les auteurs identifiés
            - Ou bien génération avec des statistiques combinées de plusieurs auteurs

        Returns :
            (void) : Au retour, la liste d'auteurs et les attributs adéquats sont configurés
        """
        if self.auteur_unique_pour_generation:
            self.prepare_single_author_generation()
            if self.liste_auteurs_pour_generation:
                PrintUtil.log_print(
                    "Utilisation simultanée de -g et -G: la génération de texte se fera pour l'auteur unique",
                      self.auteur_unique_pour_generation)
            if self.liste_auteurs_pour_generation_commune:
                PrintUtil.log_print(
                    "Utilisation simultanée de -g et -T: la génération de texte se fera pour l'auteur unique",
                      self.auteur_unique_pour_generation)
        elif self.liste_auteurs_pour_generation:
            self.prepare_multiple_author_generation()
            if self.liste_auteurs_pour_generation_commune:
                PrintUtil.log_print(
                    "Utilisation simultanée de -G et -T: la génération de texte se fera séparément par auteur")
        elif self.liste_auteurs_pour_generation_commune:
            self.prepare_fused_authors_generation()
        return

    def prepare_real_path_oeuvre(self):
        """Trouve le chemin vers les oeuvres dont les auteurs doivent être identifiés :

        Returns :
            (void) : Au retour, les chemins sont trouvés et ajoutés en attribut
        """
        # Recherche des oeuvres et ajout du path complet
        if self.oeuvre:
            real_path_oeuvre = []
            for i in range(len(self.oeuvre)):
                oeuvre = self.oeuvre[i]
                oeuvre = os.path.join(self.f_dir, oeuvre)
                real_path_file = os.path.realpath(oeuvre)
                try:
                    # https://docs.python.org/3/library/os.path.html#os.path.realpath
                    # https://www.tutorialspoint.com/python/os_readlink.htm
                    if not os.path.isfile(real_path_file):
                        if os.path.isdir(real_path_file):
                            raise IsADirectoryError(real_path_file)
                        else:
                            raise FileNotFoundError()
                    else:
                        real_path_oeuvre.append(real_path_file)
                except FileNotFoundError:
                    PrintUtil.log_print("Le fichier",
                                        real_path_file,
                                        "n'existe pas (paramètre -f ou -F)")
                    self.debug_handler.print_debug_info()
                except IsADirectoryError:
                    PrintUtil.log_print("Le nom",
                                        real_path_file,
                                        "correspond à un répertoire et non à un fichier")
                    self.debug_handler.print_debug_info()
            if len(real_path_oeuvre) == 0:
                PrintUtil.log_print("Aucune oeuvre accessible")
                self.debug_handler.print_debug_info()
                sys.exit(1)
            self.do_find_author = True
            self.oeuvre = real_path_oeuvre
        return

    def prepare_oeuvres_inconnues(self):
        """Prépare la liste des oeuvres dont l'auteur doit être identifié :

        Returns :
            (void) : Au retour, les chemins vers les oeuvres sont trouvés
        """
        if self.oeuvre_inconnue:
            self.oeuvre.append(self.oeuvre_inconnue)
            self.do_find_author = True
        if self.liste_oeuvres_inconnues:
            liste_oeuvres = os.path.join(self.f_dir, self.liste_oeuvres_inconnues)
            self.oeuvre = self.read_lines(str(liste_oeuvres))
            self.do_find_author = True
        self.prepare_real_path_oeuvre()
        return

    def block_stdout_print(self):
        """Désactive l'impression dans le code étudiant  :
            - Utilise la classe PrintUtil pour effectuer la désactivation :
                - Conserve sys.stdout pour rétablir au besoin
                - Remplace stdout par /dev/null (impression inactive)

        Returns :
            (void) : Au retour, l'impression est désactivée (si demandé)
        """
        if self.no_stdout_print_etudiant:
            PrintUtil.block_stdout()
        return

    def config_with_params(self):
        """Prépare l'instance de test pour qu'elle effectue toutes les actions requises :

        Returns :
            (void) : Au retour, l'instance de test est prête à être exécutée
        """
        self.set_log_file()
        self.block_stdout_print()
        self.prepare_dir_code()
        self.prepare_get_kth_ngram()
        self.prepare_generation()
        self.prepare_oeuvres_inconnues()
        self.do_analyze = (self.do_gen_text_multiple_auteurs
                           | self.do_find_author
                           | self.do_gen_text
                           | self.do_fused_gen_text
                           | self.do_get_kth_ngram
                           | self.do_print_auteur_distance)
        return

    def transfer_params(self, param_dict: dict) -> None:
        """Transfère tous les paramètres lus dans le fichier de configuration ou en ligne de commande :
            - Tous les paramètres rencontrés sont copiés dans l'instance de test

        Returns :
            (void) : Au retour, la liste d'auteurs est configurée
        """
        if param_dict is None:
            return
        for key, value in param_dict.items():
            if value is None or value == 'None':
                continue
            match key:
                case "d": self.dir = value
                case "dir_code": self.dir_code = value
                case "e": self.etudiants = value
                case "fichier_res": self.fichier_res = value
                case "dir_res": self.dir_res = value
                case "noPonc": self.keep_punc = False
                case "m": self.ngram_size = int(value)
                case "K": self.kth_ngram = int(value)
                case "g": self.auteur_unique_pour_generation = value
                case "G": self.liste_auteurs_pour_generation = value
                case "G_fusion": self.liste_auteurs_pour_generation_commune = value
                case "g_size": self.gen_size = int(value)
                case "g_name": self.g_name = value
                case "g_dir": self.g_dir = value
                case "f_dir": self.f_dir = value
                case "f": self.oeuvre_inconnue = value
                case "F": self.liste_oeuvres_inconnues = value
                case "timeout": self.timeout = int(value)
                case "pretty": self.beautify = bool(value)
                case "no_stdout_etudiant": self.no_stdout_print_etudiant = bool(value)
                case "compare_auteurs": self.do_print_auteur_distance = bool(value)
                case "recursion": sys.setrecursionlimit(int(value))
                case "v": self.verbose = bool(value)
                case _: continue
        return

    def read_cli(self) -> None:
        """Initialise l'objet en interprétant le fichier de configuration et la ligne de commande :
            - Lit le fichier de configuration
            - Lit la ligne de commande
            - Modifie tous les champs qui y sont définis

        Returns :
            (void) : Au retour, toutes les commandes reconnues sont comprises dans self.args
        """
        self.parse_cli()
        self.transfer_params(self.args.__dict__)
        return

    @staticmethod
    def get_sci(val: float, digits: int = 2) -> str:
        """Représente une valeur par une chaîne de caractères avec digits chiffres après le point
            - Lit le fichier de configuration
            - Lit la ligne de commande
            - Modifie tous les champs qui y sont définis

        Args:
            val (float) : La valeur en point flottant à représenter par une chaîne de caractères
            digits (int) : Le nombre de chiffres après le point à conserver

        Returns :
            tmp_str (str) : Retourne la chaîne de caractères représentant le nombre (possiblement tronqué)
        """
        if val == 0:
            return f"{val:.{digits}f}"

        tmp_str = f"{val:.{digits}E}"
        mantissa, exp = tmp_str.split("E")
        exponent = int(exp)

        if -2 <= exponent <= 2:
            tmp_str = f"{val:.{digits}f}"

        return tmp_str

    def __init__(self) -> None:
        """Constructeur pour la classe ParsingClassTestTextAn.  Initialisation de l'ensemble des éléments requis

        Args :
            (void) : Le constructeur lit :
                le fichier de configuration,
                la ligne de commande,
                et ajuste l'état de l'objet ParsingClassTextAn en conséquence

        Returns :
            (void) : Au retour, la nouvelle instance de test est prête à être utilisée
        """
        # Création des champs de base de l'instance, initialisés par la suite
        self.parser = None
        self.args = None
        self.config = None
        self.textan_module = None
        self.golden_module = None
        self.oeuvre = []
        self.analysis_result = {}
        self.auteurs = []
        self.textan = None
        self.golden = None
        self.cips = []
        self.cip = ""
        self.init_modules = {}

        # Initialisation des champs de l'instance et de leurs valeurs par défaut
        # Ces champs contrôlent l'exécution du harnais de test
        # Ces champs peuvent ensuite être modifiés
        #       par le fichier de configuration
        #       par la ligne de commande
        self.dir = "."
        self.dir_code = "."
        self.liste_auteurs_pour_generation = ""
        self.auteur_unique_pour_generation = ""
        self.liste_auteurs_pour_generation_commune = ""
        self.liste_oeuvres_inconnues = ""
        self.oeuvre_inconnue = ""
        self.liste_cips = ""
        self.etudiants = ""
        self.ngram_size = 1
        self.kth_ngram = 0
        self.keep_punc = True

        self.fichier_res = ""
        self.dir_res = ""
        self.dir_res_path = "."

        self.gen_size = 0
        self.g_name = "Gen_text_<CIP>_<AUT>_<DATE>.<HR>.<MIN>.<SEC>.txt"
        self.g_dir = "."
        self.rep_code = "."
        self.f_dir = "."

        self.do_find_author = False
        self.do_analyze = False
        self.do_gen_text = False
        self.do_fused_gen_text = False
        self.do_gen_text_multiple_auteurs = False
        self.do_get_kth_ngram = False
        self.do_print_auteur_distance = False

        self.timeout = -1
        self.beautify = True
        self.no_stdout_print_etudiant = False
        self.verbose = False

        self.read_cli()
        self.debug_handler = debug_handler_common.DebugHandler()
        self.config_with_params()

        # Initialisation des chronomètres pour mesurer et possiblement arrêter l'exécution
        self.start_time = timeit.default_timer()
        self.debug_handler.timeout = self.timeout

        return
