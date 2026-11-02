# -- Path setup --------------------------------------------------------------

import re
import sys

import snapatac2
from scverse_doc.registry import intersphinx

# -- Software version --------------------------------------------------------

# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', snapatac2.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)

# The full version, including alpha/beta/rc tags.
release = snapatac2.__version__

# pyData/Sphinx-Theme version switcher
if ".dev" in version:
    switcher_version = "dev"
else:
    switcher_version = f"{version}"

print(f'Building documentation for SnapATAC2 {release} (short version: {version}, switcher version: {switcher_version})')

# -- Project information -----------------------------------------------------

project = 'SnapATAC2'
source_repository = 'https://github.com/scverse/SnapATAC2'
source_code_directory = ''
copyright = '2022-2026, Kai Zhang'
author = 'Kai Zhang'

# -- General configuration ---------------------------------------------------

suppress_warnings = ['ref.citation']
default_role = 'code'
add_function_parentheses = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "sphinx_plotly_directive",
    # The scverse theme, source links and package registry, without scverse_doc's extension stack:
    # that would bring myst-nb, which takes the notebooks over from nbsphinx.
    "scverse_doc.theme",
    "scverse_doc.source",
    "scverse_doc.registry",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "amsmath",
    #"colon_fence",
    #"deflist",
    "dollarmath",
    #"fieldlist",
    #"html_admonition",
    #"html_image",
    #"linkify",
    #"replacements",
    #"smartquotes",
    #"strikethrough",
    #"substitution",
    #"tasklist",
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = 'bysource'
# autodoc_default_flags = ['members']
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
todo_include_todos = False

intersphinx_mapping = {
    **intersphinx(),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Configure autodoc to handle import errors gracefully
autodoc_mock_imports = []

def resolve_py_module(name):
    """Resolve Python module aliases used in documentation."""
    import sys
    if name == 'pl':
        return sys.modules.get('snapatac2.plotting')
    elif name == 'tl':
        return sys.modules.get('snapatac2.tools')
    elif name == 'pp':
        return sys.modules.get('snapatac2.preprocessing')
    elif name == 'ex':
        return sys.modules.get('snapatac2.export')
    return None

def setup(app):
    """Configure Sphinx application."""
    # Register the aliases in sys.modules for Sphinx to find them
    import snapatac2
    import sys
    
    sys.modules['snapatac2.pl'] = snapatac2.pl
    sys.modules['snapatac2.tl'] = snapatac2.tl
    sys.modules['snapatac2.pp'] = snapatac2.pp
    sys.modules['snapatac2.ex'] = snapatac2.ex

smv_branch_whitelist = r'main'  # Include all branches

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'scverse'
html_show_sphinx = False
html_show_sourcelink = False
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

html_theme_options = {
    # The docs are versioned on GitHub Pages rather than Read the Docs, so the sidebar takes pydata's switcher.
    "primary_sidebar_end": ["version-switcher"],
    "show_version_warning_banner": switcher_version == "dev",
    "switcher": {
        "version_match": switcher_version,
        "json_url": "https://raw.githubusercontent.com/scverse/SnapATAC2/refs/heads/main/docs/_static/versions.json",
    },
}
