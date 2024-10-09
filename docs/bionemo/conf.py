# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

from docutils import nodes
from sphinx import search


# -- Project information -----------------------------------------------------

project = "NVIDIA BioNeMo Framework"
copyright = "2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
author = "NVIDIA"

# The full version, including alpha/beta/rc tags
release = "v0.4.0"

# maintain left-side bar toctrees in `contents` file
# so it doesn't show up needlessly in the index page
master_doc = "contents"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "ablog",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx-prompt",
    "sphinxcontrib.bibtex",
    "sphinx_tabs.tabs",
    "sphinx_sitemap",
    # "sphinxcontrib.openapi",
    # "sphinxcontrib.redoc"
]

suppress_warnings = ["myst.domains", "ref.ref"]

numfig = True

# final location of docs for seo/sitemap
html_baseurl = "https://docs.nvidia.com/bionemo-framework/0.4.0/"  # FIXME # FIXED for now; needs to be updated at the time of product launch

# final location of the framework code repo
fw_code_baseurl = "https://registry.ngc.nvidia.com/orgs/cobwt4rder8b/containers"  # FIXME # FIXED for now; needs to be updated at the time of product launch

# final location of the framework signup/info website
bionemo_info_url = "https://www.nvidia.com/en-us/gpu-cloud/bionemo"  # FIXME

# NGC deploy info
deploy_ngc_registry = "nvcr.io"  # FIXME # FIXED for now; needs to be updated at the time of product launch
deploy_ngc_org = "nvidia"  # FIXME
deploy_ngc_org_id = "cobwt4rder8b"  # FIXME
deploy_ngc_team = "clara"  # FIXME
deploy_container_name = "bionemo-framework"  # FIXME
deploy_container_tag = "1.3"  # FIXME

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    # "html_image",
    "colon_fence",
    # "smartquotes",
    "replacements",
    "substitution",
    # "linkify",
]
myst_heading_anchors = 4

# redoc = [
#     {
#         'name': 'NeMo LLM API',
#         'page': 'api',
#         'spec': 'specs/completions.yml',
#     }
# ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "perf_data/perf_data_build_scripts/README.md",
    "ngc-artifacts-desc/*",
    "tutorials/README.md",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Refer to the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "_static/nvidia-logo-horiz-rgb-blk-for-screen.png"
html_title = "NVIDIA BioNeMo Framework"
html_short_title = "BioNeMo Framework"
html_copy_source = True
html_sourcelink_suffix = ""
html_favicon = "_static/nvidia-logo-vert-rgb-blk-for-screen.png"
html_last_updated_fmt = ""
html_additional_files = ["index.html"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "path_to_docs": "docs",
    # "launch_buttons": {
    #     "binderhub_url": "https://mybinder.org",
    #     "colab_url": "https://colab.research.google.com/",
    #     "deepnote_url": "https://deepnote.com/",
    #     "notebook_interface": "jupyterlab",
    #     "thebe": True,
    #     # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
    # },
    "use_edit_page_button": False,
    "use_issues_button": False,
    "use_repository_button": False,
    "use_download_button": False,
    "logo_only": False,
    "show_toc_level": 4,
    "extra_navbar": "",
    "extra_footer": "",
}

version_short = release
myst_substitutions = {
    "version_num": version_short,
    "bionemo_info_url": f"[information website]({bionemo_info_url})",
    "model_license_slug": "Apache License",
    "deploy_ngc_registry": f"{deploy_ngc_registry}",
    "deploy_ngc_org_team": f"{deploy_ngc_org}/{deploy_ngc_team}" if deploy_ngc_team else deploy_ngc_org,
    "deploy_ngc_org": f"{deploy_ngc_org}",
    "deploy_ngc_team": f"{deploy_ngc_team}" if deploy_ngc_team else "",
    "deploy_container_name": f"{deploy_container_name}",
    "deploy_container_tag": f"{deploy_container_tag}",
}


def ultimateReplace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result


# this is a necessary hack to allow us to fill in variables that exist in code blocks
ultimate_replacements = {
    "{version_num}": version_short,
    "{deploy_ngc_registry}": f"{deploy_ngc_registry}",
    "{deploy_ngc_org_team}": f"{deploy_ngc_org}/{deploy_ngc_team}" if deploy_ngc_team else deploy_ngc_org,
    "{deploy_ngc_org}": f"{deploy_ngc_org}",
    "{deploy_ngc_team}": f"{deploy_ngc_team}" if deploy_ngc_team else "",
    "{deploy_container_name}": f"{deploy_container_name}",
    "{deploy_container_tag}": f"{deploy_container_tag}",
}

bibtex_bibfiles = ["references.bib"]
# To test that style looks good with common bibtex config
bibtex_reference_style = "author_year"
bibtex_default_style = "unsrt"

### We currrently use Myst: https://myst-nb.readthedocs.io/en/latest/use/execute.html
nb_execution_mode = "off"  # Global execution disable
# execution_excludepatterns = ['tutorials/tts-python-basics.ipynb']  # Individual notebook disable


def inject_uniprot_visuals(app, pagename, templatename, context, doctree):
    """
    Inject specific JavaScript and CSS files for the uniprot page to ensure that
    js/functionality specific to this page do not affect other pages in the documentation.
    """
    uniprot_url = "datasets/uniprot"

    # only inject js/css to uniprot page
    if pagename == uniprot_url:
        base_path = "_static/uniprot_visual"
        js_files = ["d3.min.js", "unirefClusterSample.js", "UniprotCirclePackViz.js", "main.js"]
        css_files = ["uniprot_styles.css"]

        # inject js -- note, app.add_js_file inside html-page-context doesn't work,
        # so here I directly append our scripts alongside the uniprot page header scripts
        context["script_files"].extend(f"{base_path}/js/{file}" for file in js_files)

        # inject css
        context["css_files"].extend(f"{base_path}/css/{file}" for file in css_files)


def setup(app):
    app.add_config_value("ultimate_replacements", {}, True)
    app.connect("source-read", ultimateReplace)
    app.add_js_file("https://js.hcaptcha.com/1/api.js")

    visitor_script = "//assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js"

    if visitor_script:
        app.add_js_file(visitor_script)

    # add uniprot visual to uniprot page
    app.connect("html-page-context", inject_uniprot_visuals)

    # if not os.environ.get("READTHEDOCS") and not os.environ.get("GITHUB_ACTIONS"):
    #     app.add_css_file(
    #         "https://assets.readthedocs.org/static/css/readthedocs-doc-embed.css"
    #     )
    #     app.add_css_file("https://assets.readthedocs.org/static/css/badge_only.css")

    #     # Create the dummy data file so we can link it
    #     # ref: https://github.com/readthedocs/readthedocs.org/blob/bc3e147770e5740314a8e8c33fec5d111c850498/readthedocs/core/static-src/core/js/doc-embed/footer.js  # noqa: E501
    #     app.add_js_file("rtd-data.js")
    #     app.add_js_file(
    #         "https://assets.readthedocs.org/static/javascript/readthedocs-doc-embed.js",
    #         priority=501,
    #     )


# Patch for sphinx.search stemming short terms (that is, tts -> tt)
# https://github.com/sphinx-doc/sphinx/blob/4.5.x/sphinx/search/__init__.py#L380
def sphinxSearchIndexFeed(self, docname: str, filename: str, title: str, doctree: nodes.document):
    """Feed a doctree to the index."""
    self._titles[docname] = title
    self._filenames[docname] = filename

    visitor = search.WordCollector(doctree, self.lang)
    doctree.walk(visitor)

    # memoize self.lang.stem
    def stem(word: str) -> str:
        try:
            return self._stem_cache[word]
        except KeyError:
            self._stem_cache[word] = self.lang.stem(word).lower()
            return self._stem_cache[word]

    _filter = self.lang.word_filter

    for word in visitor.found_title_words:
        stemmed_word = stem(word)
        if len(stemmed_word) > 3 and _filter(stemmed_word):
            self._title_mapping.setdefault(stemmed_word, set()).add(docname)
        elif _filter(word):  # stemmer must not remove words from search index
            self._title_mapping.setdefault(word.lower(), set()).add(docname)

    for word in visitor.found_words:
        stemmed_word = stem(word)
        # again, stemmer must not remove words from search index
        if len(stemmed_word) <= 3 or not _filter(stemmed_word) and _filter(word):
            stemmed_word = word.lower()
        already_indexed = docname in self._title_mapping.get(stemmed_word, set())
        if _filter(stemmed_word) and not already_indexed:
            self._mapping.setdefault(stemmed_word, set()).add(docname)


search.IndexBuilder.feed = sphinxSearchIndexFeed
