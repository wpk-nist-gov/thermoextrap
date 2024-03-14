#
# python_boilerplate documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
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
"""Build docs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path("../src").resolve()))

import thermoextrap

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "autodocsumm",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    # "IPython.sphinxext.ipython_directive",
    # "IPython.sphinxext.ipython_console_highlighting",
    # "nbsphinx",
    # - easier external links
    # "sphinx.ext.extlinks",
    # - view source code on created page
    # "sphinx.ext.viewcode",
    # - view source code on github
    "sphinx.ext.linkcode",
    # - add copy button
    "sphinx_copybutton",
    # - redirect stuff?
    # "sphinxext.rediraffe",
    # - pretty things up?
    # "sphinx_design"
    # - myst stuff
    "myst_nb",
    # "myst_parser",
]

nitpicky = True
autosectionlabel_prefix_document = True

# -- myst stuff ---------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    # "linkify",
    "strikethrough",
    "substitution",
    "tasklist",
    # "attrs_inline",
    # "attrs_block",
]


myst_heading_anchors = 3
myst_footnote_transition = True
myst_dmath_double_inline = True
myst_enable_checkboxes = True
myst_substitutions = {
    "role": "[role](#syntax/roles)",
    "directive": "[directive](#syntax/directives)",
}
# myst_enable_extensions = [
#     "dollarmath",
#     "amsmath",
#     "deflist",
#     # "html_admonition",
#     "html_image",
#     "colon_fence",
#     # "smartquotes",
#     # "replacements",
#     # "linkify",
#     # "substitution",
#     "attrs_inline",
#     "attrs_block",
# ]

myst_url_schemes = ("http", "https", "mailto")

nb_execution_mode = "off"
# nb_execution_mode = "cache"
# nb_execution_mode = "auto"

# set the kernel name
nb_kernel_rgx_aliases = {
    "thermoextrap.*": "python3",
    "conda.*": "python3",
}

nb_execution_allow_errors = True

# Whether to remove stderr
nb_output_stderr = "remove"

# - top level variables --------------------------------------------------------
# set github_username variable to be subbed later.
# this makes it easy to switch from wpk -> usnistgov later
github_username = "usnistgov"

html_context = {
    "github_user": github_username,
    "github_repo": "thermoextrap",
    "github_version": "main",
    "doc_path": "docs",
}

# -- python3 ---------------------------------------------------------------
autosummary_generate = True
# autosummary_generate = False
autodoc_member_order = "bysource"

# autosummary_ignore_module_all = False
# autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_flags = [
    # Make sure that any autodoc declarations show the right members
    "members",
    "inherited-members",
    "private-members",
    "show-inheritance",
]
autodoc_typehints = "none"

# -- napoleon ------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "hashable": ":term:`hashable`",
    "file-like": ":term:`file-like <file-like object>`",
    # special terms
    # "same type as caller": "*same type as caller*",  # does not work, yet
    # "same type as values": "*same type as values*",  # does not work, yet
    # stdlib type aliases
    "MutableMapping": "~collections.abc.MutableMapping",
    "sys.stdout": ":obj:`sys.stdout`",
    "timedelta": "~datetime.timedelta",
    "string": ":class:`string <str>`",
    # numpy terms
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    "scalar": ":term:`scalar`",
    "array": ":term:`array`",
    # matplotlib terms
    "color-like": ":py:func:`color-like <matplotlib.colors.is_color_like>`",
    "matplotlib colormap name": ":doc:`matplotlib colormap name <matplotlib:gallery/color/colormap_reference>`",
    "matplotlib axes object": ":py:class:`matplotlib axes object <matplotlib.axes.Axes>`",
    "colormap": ":py:class:`colormap <matplotlib.colors.Colormap>`",
    # objects without namespace: xarray
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "Variable": "~xarray.Variable",
    "DatasetGroupBy": "~xarray.core.groupby.DatasetGroupBy",
    "DataArrayGroupBy": "~xarray.core.groupby.DataArrayGroupBy",
    "CentralMoments": "~cmomy.CentralMoments",
    "xCentralMoments": "~cmomy.xCentralMoments",
    # objects without namespace: numpy
    "ndarray": "~numpy.ndarray",
    "MaskedArray": "~numpy.ma.MaskedArray",
    "dtype": "~numpy.dtype",
    "ComplexWarning": "~numpy.ComplexWarning",
    # objects without namespace: pandas
    "Index": "~pandas.Index",
    "MultiIndex": "~pandas.MultiIndex",
    "CategoricalIndex": "~pandas.CategoricalIndex",
    "TimedeltaIndex": "~pandas.TimedeltaIndex",
    "DatetimeIndex": "~pandas.DatetimeIndex",
    "Series": "~pandas.Series",
    "DataFrame": "~pandas.DataFrame",
    "Categorical": "~pandas.Categorical",
    "Path": "~~pathlib.Path",
    # objects with abbreviated namespace (from pandas)
    "pd.Index": "~pandas.Index",
    "pd.NaT": "~pandas.NaT",
    "Expr": "~sympy.core.expr.Expr",
    "Symbol": "~sympy.core.symbol.Symbol",
    "symFunction": "~sympy.core.function.Function",
    "ExtrapModel": "~thermoextrap.models.ExtrapModel",
    "Generator": "~numpy.random.Generator",
    # "SeedSequence": "~numpy.random.SeedSequence",
    # "BitGenerator": "~numpy.random.BitGenerator",
    # "DataCentralMoments": "~cmomy.DataCentralMoments"
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "thermoextrap"
copyright = "2020, William P. Krekelberg"  # noqa: A001
author = "William P. Krekelberg"


# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
def _get_version() -> str:
    if (version := os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION")) is None:
        version = thermoextrap.__version__
    return version


release = version = _get_version()


# if always want to print "latest"
# release = "latest"
# version = "latest"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "jupyter_execute",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

html_theme_options = {
    # "analytics_id": ''  this is configured in rtfd.io
    # "canonical_url": "",
    "repository_url": f"https://github.com/{github_username}/thermoextrap",
    "repository_branch": html_context["github_version"],
    "path_to_docs": html_context["doc_path"],
    # "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "home_page_in_toc": True,
    "show_toc_level": 3,
    "show_navbar_depth": 0,
}
# handle nist css/js from here.
html_css_files = [
    # "css/nist-combined.css",
    "https://pages.nist.gov/nist-header-footer/css/nist-combined.css",
    "https://pages.nist.gov/leaveNotice/css/jquery.leaveNotice.css",
]

html_js_files = [
    "https://code.jquery.com/jquery-3.6.2.min.js",
    "https://pages.nist.gov/nist-header-footer/js/nist-header-footer.js",
    # "js/nist-header-footer.js",
    "https://pages.nist.gov/leaveNotice/js/jquery.leaveNotice-nist.min.js",
    "js/leave_notice.js",
    # google stuff:
    (
        "https://dap.digitalgov.gov/Universal-Federated-Analytics-Min.js?agency=NIST&subagency=github&pua=UA-66610693-1&yt=true&exts=ppsx,pps,f90,sch,rtf,wrl,txz,m1v,xlsm,msi,xsd,f,tif,eps,mpg,xml,pl,xlt,c",
        {"async": "async", "id": "_fed_au_ua_tag", "type": "text/javascript"},
    ),
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Sometimes the savefig directory doesn't exist and needs to be created
# https://github.com/ipython/ipython/issues/8733
# becomes obsolete when we can pin ipython>=5.2; see ci/requirements/doc.yml
def _get_ipython_savefig_dir() -> str:
    d = Path(__file__).parent / "_build" / "html" / "_static"
    if not d.is_dir():
        d.mkdir(parents=True)
    return str(d)


ipython_savefig_dir = _get_ipython_savefig_dir()

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
today_fmt = "%Y-%m-%d"
html_last_updated_fmt = today_fmt


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "thermoextrapdoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "thermoextrap.tex",
        "thermoextrap Documentation",
        "William P. Krekelberg",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "thermoextrap",
        "thermoextrap Documentation",
        [author],
        1,
    ),
]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "thermoextrap",
        "thermoextrap Documentation",
        author,
        "thermoextrap",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- user defined stuff ------------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest", None),
    # "matplotlib": ("https://matplotlib.org", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "cftime": ("https://unidata.github.io/cftime", None),
    "sparse": ("https://sparse.pydata.org/en/latest/", None),
    "cmomy": ("https://pages.nist.gov/cmomy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "gpflow": ("https://gpflow.github.io/GPflow/develop/", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",
    ),
    "tensorflow_probability": (
        "https://www.tensorflow.org/probability/api_docs/python",
        "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tfp_py_objects.inv",
    ),
}

linkcheck_ignore = ["https://doi.org/"]


# based on numpy doc/source/conf.py
def linkcode_resolve(domain: str, info: dict[str, Any]) -> str | None:
    """Determine the URL corresponding to Python object."""
    import inspect
    from operator import attrgetter

    if domain != "py":
        return None

    parent_name, *sub_parts = info["module"].split(".")
    parent_mod = sys.modules.get(parent_name)

    try:
        if len(sub_parts) > 0:
            sub_name = ".".join(sub_parts)
            obj = attrgetter(sub_name)(parent_mod)
        else:
            obj = parent_mod

        # get fullname
        obj = attrgetter(info["fullname"])(obj)

    except AttributeError:
        return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    # fmt: off
    fn = os.path.relpath(fn, start=Path(thermoextrap.__file__).parent)
    # fmt: on

    return f"https://github.com/{github_username}/thermoextrap/blob/{html_context['github_version']}/src/thermoextrap/{fn}{linespec}"


# only set spelling stuff if installed:
try:
    import sphinxcontrib.spelling  # noqa: F401

    extensions += ["sphinxcontrib.spelling"]
    spelling_word_list_filename = "spelling_wordlist.txt"

except ImportError:
    pass
