# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------

project = 'Problématique_GIF270'
copyright = '2018-2026 Frédéric Mailhot'
author = 'Frédéric Mailhot'

# The full version, including alpha/beta/rc tags
release = '1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx_pyreverse']
#extensions = ['sphinxcontrib.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx_pyreverse']
#extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx_pyreverse']

sphinx_pyreverse_all_ancestors = "y"
sphinx_pyreverse_module_names = "y"
sphinx_pyreverse_show_associated = "1"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}


# Modification de certains paramètres:
# Langue par défaut:  français  (pour le mois de la date)
# Format de la date (today): jour mois année
language='fr'
today_fmt='%d %b %Y'
autoclass_content = 'both'
