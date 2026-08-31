import sys
import os
import sphinx_rtd_theme
project = 'QMLearn'
copyright = '2022, Pavanello Research Group'
author = 'Pavanello Research Group'
release = '0.0.1'

sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.youtube",
    "nbsphinx",
    'sphinx_panels',
]

templates_path = ['templates']
numpydoc_show_class_members = True

source_suffix = '.rst'

master_doc = 'index'

nbsphinx_execute = 'never'
language = None

exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']



html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_bootstrap_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_favicon = 'static/qmlearn.ico'
html_logo = 'static/qmlearn.png'
html_style = 'custom.css'
html_static_path = ['static']
html_last_updated_fmt = '%A, %d %b %Y %H:%M:%S'

html_theme_options = {
    'prev_next_buttons_location': 'both',
}

latex_show_urls = 'inline'
latex_show_pagerefs = True
latex_documents = [('index', not True)]


#Add external links to source code
def linkcode_resolve(domain, info):
    print('info module', info)
    if domain != 'py' or not info['module']:
        return None

    filename = info['module'].replace('.', '/')+'.py'
    return "https ://gitlab.com/pavanello-research-group/qmlearn/tree/master/%s" % filename

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True




