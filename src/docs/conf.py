"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-09-27 10:20:11
Last Revision  : 2023-09-27 10:32:59
Last Author    : Jie Li
File Path      : /AutoCPD/src/docs/conf.py
Description    :








Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "AutoCPD"
copyright = "2023, Jie Li"
author = "Jie Li"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
