# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Paspailleur'
copyright = '2025, SmartFCA/LORIA'
author = 'Egor Dudyrev'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_design',
    'myst_nb'
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_member_order = 'bysource'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_sidebars = { # this is for the primary sidebar (the one on the left)
    "**": ["sidebar-nav-bs"],
    "index": [],
    "example_from_titanic": [],
}

html_theme_options = { # this is for the secondary sidebar (the one on the right side)
    "show_toc_level": 3,
    "github_url": "https://github.com/smartFCA/paspailleur.git",
    "secondary_sidebar_items": ["page-toc", "sourcelink"]
}

nb_execution_timeout = 60 * 10
nb_execution_mode = 'cache'
