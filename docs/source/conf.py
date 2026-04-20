# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Idefix2Python"
copyright = "2026, David Fang"
author = "David Fang"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Pulls docstrings from code
    "sphinx.ext.napoleon",  # Supports Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Adds links to the source code
    "sphinx.ext.githubpages",  # Useful for hosting on GitHub
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinxcontrib.video",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
