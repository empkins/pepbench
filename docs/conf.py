# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import inspect

# -- Path setup --------------------------------------------------------------
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import toml

__location__ = os.path.join(os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe())))
HERE = Path(__file__)

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.join(__location__, "../src/pepbench"))


URL = "https://github.com/empkins/pepbench"

# -- Project information -----------------------------------------------------

# Info from uv config:
info = toml.load("../pyproject.toml")["project"]

project = info["name"]
author = ", ".join([k["name"] for k in info["authors"]])
release = info["version"]

copyright = (
    f"2024 - {datetime.now().year}, Machine Learning and Data Analytics (MaD) Lab, Friedrich-Alexander-Universität "
    "Erlangen-Nürnberg (FAU)"
)

# -- Copy the README and Changelog and fix image path --------------------------------------
HERE = Path(__file__).parent
EXAMPLE_NOTEBOOKS_DIR = HERE.joinpath("examples/_notebooks")

with (HERE.parent / "README.md").open() as f:
    out = f.read()
with (HERE / "README.md").open("w+") as f:
    f.write(out)

with (HERE.parent / "CHANGELOG.md").open() as f:
    out = f.read()
with (HERE / "CHANGELOG.md").open("w+") as f:
    f.write(out)


def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if os.path.isfile(os.path.join(dir, c)) and (not c.endswith(".ipynb")):
            result += [c]
    return result


subprocess.run(["python", "-m", "ipykernel", "install", "--user", "--name", "pepbench"], check=True)

shutil.rmtree(EXAMPLE_NOTEBOOKS_DIR, ignore_errors=True)
shutil.copytree(HERE.parent.joinpath("examples"), EXAMPLE_NOTEBOOKS_DIR, ignore=all_but_ipynb)
for file in EXAMPLE_NOTEBOOKS_DIR.glob("*.ipynb"):
    with file.open() as f:
        out = f.read()
    out = out.replace("%matplotlib widget", "%matplotlib inline")
    with file.open("w+") as f:
        f.write(out)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosummary",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    # "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "button",
    "nbsphinx",
    # "sphinx_rtd_theme",
]

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

# Taken from sklearn config
# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/" "tex-chtml.js"

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "special_members": True,
}
# autodoc_typehints = 'description'  # Does not work as expected. Maybe try at future date again
autodoc_typehints = "signature"

# autodoc_inherit_docstrings = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = True

# This value selects if automatically documented members are sorted alphabetical (value 'alphabetical'),
# by member type (value 'groupwise') or by source order (value 'bysource'). The default is alphabetical.
autodoc_member_order = "bysource"

# This value selects what content will be inserted into the main body of an autoclass directive.
autoclass_content = "class"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "templates"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# Enable markdown
extensions.append("recommonmark")

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Activate the theme.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": URL,
    "show_prev_next": False,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ["button.css"]

# -- Options for extensions --------------------------------------------------

# intersphinx configuration
intersphinx_module_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "python": ("https://docs.python.org/3/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "pyscaffold": ("https://pyscaffold.org/en/stable", None),
    "biopsykit": ("https://biopsykit.readthedocs.io/en/latest/", None),
    # "neurokit2": ("https://neurokit2.readthedocs.io/en/latest", None),
    # "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "nilspodlib": ("https://nilspodlib.readthedocs.io/en/latest/", None),
    "pingouin": ("https://pingouin-stats.org/", None),
}

user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:25.0) Gecko/20100101 Firefox/25.0"

# Sphinx Gallary
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["./auto_examples"],
    "reference_url": {"pepbench": None, **{k: v[0] for k, v in intersphinx_module_mapping.items()}},
    # 'default_thumb_file': 'fig/logo.png',
    "backreferences_dir": "modules/generated/backreferences",
    "doc_module": ("pepbench",),
    "filename_pattern": re.escape(os.sep),
    "remove_config_comments": True,
    "show_memory": True,
}


from sphinxext.githublink import make_linkcode_resolve

linkcode_resolve = make_linkcode_resolve(
    "pepbench",
    "https://github.com/empkins/pepbench/blob/{revision}/{package}/{path}#L{lineno}",
)

nbsphinx_epilog = r"""

{% set docname = env.doc2path(env.docname) %}


.. button::
   :text: Download Notebook</br>(Right-Click -> Save Link As...)
   :link: https://raw.githubusercontent.com/mad-lab-fau/BioPsyKit/main/examples/{{ docname.split('/')[-1] }}

"""

# -- External mapping --------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pyscaffold": ("https://pyscaffold.org/en/stable", None),
    "biopsykit": ("https://biopsykit.readthedocs.io/en/latest/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "nilspodlib": ("https://nilspodlib.readthedocs.io/en/latest/", None),
}
