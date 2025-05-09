#!/usr/bin/env python
#
# lime_tbx documentation build configuration file, created by
# cookiecutter
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys
import re

from sphinx.application import Sphinx

import lime_tbx

sys.path.append(os.path.abspath("_templates"))

sys.path.insert(0, os.path.abspath(".."))

# SH added to run apidoc on build
this_directory = os.path.dirname(__file__)


def run_apidoc(_):
    ignore_paths = ["./../../*/tests/"]

    argv = [
        "-f",
        "-T",
        "-e",
        "-M",
        "-o",
        os.path.join(this_directory, "content", "api"),
        lime_tbx.__path__[0],
    ] + ignore_paths

    try:
        # Sphinx 1.7+
        from sphinx.ext import apidoc

        apidoc.main(argv)
    except ImportError:
        # Sphinx 1.6 (and earlier)
        from sphinx import apidoc

        argv.insert(0, apidoc.__file__)
        apidoc.main(argv)


from docutils.nodes import Text, reference, image, raw


def process_node(node):
    if isinstance(node, Text):
        pass
    elif isinstance(node, reference):
        # Modifica atributos como 'refuri' para enlaces relativos
        starts = ("./docs", "../")
        if node.get("refuri", "").startswith(starts):
            node["refuri"] = "../_static" + node["refuri"][node["refuri"].rindex("/") :]
        if node.get("refid", "").startswith(starts):
            node["refid"] = "../_static" + node["refid"][node["refid"].rindex("/") :]
    elif isinstance(node, image):
        pass
    elif isinstance(node, raw):
        if node.get("format") == "html":
            raw_content = node.astext()
            # Traverse sibling nodes to gather additional content
            sibling = node.next_node(descend=False, ascend=False)
            while (
                sibling and isinstance(sibling, raw) and sibling.get("format") == "html"
            ):
                raw_content += sibling.astext()
                sibling = sibling.next_node(descend=False, ascend=False)
            # Replace relative links (Markdown-style and HTML-style)
            updated_content = re.sub(
                r"\((\.+\/((\.\.\/)*)(((?!images|_static)[^)\/])*\/)*((?:images|_static)\/)?([^)]+))\)",
                r"(../\2_static/\7)",
                raw_content,
            )
            updated_content = re.sub(
                r'<a\s+href="(\.+\/((\.\.\/)*)(((?!images|_static)[^"\/])*\/)*((?:images|_static)\/)?([^"]+))"',
                r'<a href="../\2_static/\7"',
                updated_content,
            )
            updated_content = re.sub(
                r'<img\s+src="(\.+\/((\.\.\/)*)(((?!images|_static)[^"\/])*\/)*((?:images|_static)\/)?([^"]+))"',
                r'<img src="../\2_static/\7"',
                updated_content,
            )
            # Now replace the links that should go to a content html page
            # The `html` is included in the pattern because it will be overwritten
            # by the previouse lines and we want it to be affected again by these lines.
            updated_content = re.sub(
                r"\(\.\.\/_static\/([^)]+)\.(md|rst|html)\)",
                r"(./\1.html)",
                updated_content,
            )
            updated_content = re.sub(
                r'<a\s+href="\.\.\/_static\/([^"]+)\.(md|rst|html)"',
                r'<a href="./\1.html"',
                updated_content,
            )

            if updated_content != raw_content:
                node.rawsource = updated_content  # Update the first raw node
                node.children = []  # Clear any children nodes
                node.append(raw("", updated_content, format="html"))
    for chnode in node.children:
        process_node(chnode)


def process_links_in_doctree(app, doctree, docname):
    """
    Modifica los enlaces relativos en el doctree antes de que se genere el HTML.
    """
    if docname in (
        "index",
        "content/readme",
        "content/design",
        "content/implementation",
        "content/user_guide/overview",
        "content/user_guide/installation",
        "content/user_guide/configuration",
        "content/user_guide/simulations",
        "content/user_guide/comparisons",
        "content/user_guide/coefficients",
    ):  # Aplica cambios solo a 'index.rst' o el docname relevante
        for node in doctree.traverse():
            process_node(node)


def on_builder_inited(app: Sphinx):
    if app.builder.name not in ("pdf", "latex"):
        run_apidoc(app)


def setup(app: Sphinx):
    app.connect("builder-inited", on_builder_inited)
    app.connect("doctree-resolved", process_links_in_doctree)


project_title = "lime_tbx".replace("_", " ").title()

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Attempt to make links automatially
default_role = "code"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
# CFAB added napolean to support google-style docstrings
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

myst_enable_extensions = [
    "tasklist",  # Enables checkbox rendering
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "linkify",
    "colon_fence",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_parsers = {
    ".md": "myst_parser.sphinx_",
}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.

IS_LATEX = "html" not in sys.argv and ("latex" in sys.argv or "pdf" in sys.argv)

master_doc = "index"
if IS_LATEX:
    master_doc = "index_pdf"

# General information about the project.
project = project_title
author = "Javier Gatón, Pieter De Vis, Stefan Adriaensen, Jacob Fahy, Ramiro González Catón, Carlos Toledano, África Barreto, Agnieszka Bialek, Marc Bouvet"
copyright = "2024, European Space Agency (ESA). Code and documentation licensed under the GNU Lesser General Public License (LGPL), Version 3.0"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = ".".join(lime_tbx.__version__.split(".")[:2])
# The full version, including alpha/beta/rc tags.
release = lime_tbx.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "images"]

# SH added to override wide tables in RTD theme
# html_context = {
#    "css_files": [
#        "_static/theme_overrides.css",
#    ],
# }

html_css_files = [
    "theme_overrides.css",
]

numfig = True
numfig_secnum_depth = 0  # Disable section-based numbering
numfig_format = {
    "figure": "<i>Figure %s:</i>",
    "table": "Table %s:",
}
if IS_LATEX:
    numfig_format = {
        "figure": "Figure %s",
        "table": "Table %s",
    }


html_favicon = "favicon.png"

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "lime_tbxdoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    "extrapackages": r"""
\usepackage{textcomp}
\usepackage{newunicodechar}
\newunicodechar{⁻}{\textsuperscript{-}}
\newunicodechar{¹}{\ifmmode{}^1\else\textonesuperior\fi}
\newunicodechar{²}{\ifmmode{}^2\else\texttwosuperior\fi}
\newunicodechar{³}{\ifmmode{}^3\else\textthreesuperior\fi}
""",
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    "preamble": r"""
\usepackage{graphicx}
\graphicspath{{./_static/}}
\usepackage{caption}
\captionsetup[figure]{labelfont=it, textfont=normalfont}
""",
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_title = "LIME Toolbox User Guide"
latex_author = author
if "," in latex_author:
    latex_author = [aa.strip() for aa in latex_author.split(",")]
    latex_author = [
        f'{" ".join(aa.split()[1:])}, {aa.split()[0]}' for aa in latex_author
    ]
    latex_author = r"\and ".join(latex_author)
latex_documents = [
    (
        master_doc,
        "user_manual.tex",
        latex_title,
        latex_author,
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "lime_tbx", "lime_tbx Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "lime_tbx",
        latex_title,
        latex_author,
        "lime_tbx",
        "The LIME TBX is a Python package providing a comprehensive toolbox for utilizing the LIME (Lunar Irradiance Model of ESA) model to simulate lunar observations and compare them with remote sensing data of the Moon.",
        "Miscellaneous",
    ),
]
