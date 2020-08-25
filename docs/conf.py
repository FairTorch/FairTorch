import sys, os
import mock

sys.path.insert(0, os.path.abspath('./torch_stubs'))
import nn
import optim
import functional as F

MOCK_MODULES = ['torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'matplotlib', 'matplotlib.pyplot']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock(Module=object, 
                                      relu=F.relu, 
                                      MSELoss=nn.MSELoss,
                                      CrossEntropyLoss=nn.CrossEntropyLoss,
                                      Adam=optim.Adam
                                      )

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'FairTorch'
copyright = '2020, FairTorch Team'
author = 'FairTorch Team'

# The full version, including alpha/beta/rc tags
release = 'b0.0.1'


# -- General configuration ---------------------------------------------------

master_doc = 'index'

html_favicon = 'fairtorchlogo.png'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc'
]

intersphinx_mapping = {
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
#    'style_nav_header_background' : 'green'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# add sourcecode to path
sys.path.insert(0, os.path.abspath('../fairtorch'))
sys.path.insert(0, os.path.abspath('../fairtorch/preprocessing'))
sys.path.insert(0, os.path.abspath('../fairtorch/training'))
sys.path.insert(0, os.path.abspath('../fairtorch/evaluation'))
