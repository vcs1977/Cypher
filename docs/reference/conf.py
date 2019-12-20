# -*- coding: utf-8 -*-
#

import sys
import os

sys.path.insert(0, os.path.abspath("../../"))

extensions = ["sphinx.ext.autodoc", "sphinx_rtd_theme", "sphinx.ext.doctest", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
source_suffix = ".rst"
master_doc = "index"
project = u"Cypher"
copyright = u"Tanish Shinde, simon.blanke@yahoo.com"
exclude_patterns = ["_build"]
pygments_style = "sphinx"
html_theme = "default"
autoclass_content = "both"
html_theme = "sphinx_rtd_theme"
html_sidebars = {}
