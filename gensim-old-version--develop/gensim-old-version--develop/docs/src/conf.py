# -*- coding: utf-8 -*-
#
# gensim documentation build configuration file, created by
# sphinx-quickstart on Wed Mar 17 13:42:21 2010.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration -----------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinxcontrib.napoleon',
    'sphinx.ext.imgmath',
    'sphinxcontrib.programoutput',
    'sphinx_gallery.gen_gallery',
]
autoclass_content = "both"

napoleon_google_docstring = False  # Disable support for google-style docstring

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'indextoc'

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {'index': './_templates/indexcontent.html'}

# General information about the project.
project = u'gensim'
copyright = u'2009-now'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '4.3.1.dev0'
# The full version, including alpha/beta/rc tags.
release = '4.3.1.dev0'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
# unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = ['_build']

exclude_patterns = ['gallery/README.rst']

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
# html_theme = 'default'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# main_colour = "#ffbbbb"

html_theme_options = {
# "rightsidebar": "false",
# "stickysidebar": "true",
# "bodyfont": "'Lucida Grande', 'Lucida Sans Unicode', 'Geneva', 'Verdana', 'sans-serif'",
# "headfont": "'Lucida Grande', 'Lucida Sans Unicode', 'Geneva', 'Verdana', 'sans-serif'",
# "sidebarbgcolor": "fuckyou",
# "footerbgcolor": "#771111",
# "relbarbgcolor": "#993333",
# "sidebartextcolor": "#000000",
# "sidebarlinkcolor": "#330000",
# "codebgcolor": "#fffff0",
# "headtextcolor": "#000080",
# "headbgcolor": "#f0f0ff",
# "bgcolor": "#ffffff",
}


# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['.']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "gensim"

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = ''

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
# html_css_files = [
#     'erp/css/global.css',
#     'erp/css/structure.css',
#     'erp/css/erp2.css',
#     'erp/css/custom.css',
# ]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {}  # {'index': ['download.html', 'globaltoc.html', 'searchbox.html', 'indexsidebar.html']}
# html_sidebars = {'index': ['globaltoc.html', 'searchbox.html']}

# If false, no module index is generated.
# html_use_modindex = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

html_domain_indices = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'gensimdoc'

html_show_sphinx = False

# -- Options for LaTeX output --------------------------------------------------

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', 'gensim.tex', u'gensim Documentation', u'Radim Řehůřek', 'manual')]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = False

# Additional stuff for the LaTeX preamble.
# latex_preamble = ''

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_use_modindex = True

suppress_warnings = ['image.nonlocal_uri', 'ref.citation', 'ref.footnote']


def sort_key(source_dir):
    """Sorts tutorials and guides in a predefined order.

    If the predefined order doesn't include the filename we're looking for,
    fallback to alphabetical order.
    """
    core_order = [
        'run_core_concepts.py',
        'run_corpora_and_vector_spaces.py',
        'run_topics_and_transformations.py',
        'run_similarity_queries.py',
    ]

    tutorials_order = [
        'run_word2vec.py',
        'run_doc2vec_lee.py',
        'run_fasttext.py',
        'run_annoy.py',
        'run_lda.py',
        'run_wmd.py',
        'run_summarization.py',
    ]

    howto_order = [
        'run_downloader_api.py',
        'run_binder.py',
        'run_doc.py',
        'run_doc2vec_imdb.py',
        'run_news_classification.py',
        'run_compare_lda.py',
    ]

    order = core_order + tutorials_order + howto_order
    files = sorted(os.listdir(source_dir))

    def key(arg):
        try:
            return order.index(arg)
        except ValueError:
            return files.index(arg)

    return key


import sphinx_gallery.sorting
sphinx_gallery_conf = {
    'examples_dirs': 'gallery',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path where to save gallery generated examples
    'show_memory': True,
    'filename_pattern': 'run',
    'subsection_order': sphinx_gallery.sorting.ExplicitOrder(
        [
            'gallery/core',
            'gallery/tutorials',
            'gallery/howtos',
            'gallery/other',
        ],
    ),
    'within_subsection_order': sort_key,
}
