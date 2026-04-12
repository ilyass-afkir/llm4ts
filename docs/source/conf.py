# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

sys.path.insert(0, os.path.abspath('../../src'))

project = 'Ilyass Afkir'
copyright = '2026, Ilyass Afkir'
author = 'Ilyass Afkir'
release = '1.0'
html_title = "Master Thesis Code Documentation"
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # auto-generate docs from docstrings
    'sphinx.ext.viewcode',    # links to source code
    'sphinx.ext.napoleon',    # Google/NumPy style docstrings
]

templates_path = ['_templates']
exclude_patterns = []
autodoc_mock_imports = ["pyrosm"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

