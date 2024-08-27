# Useful Tips on How to Build the Documentation

## Steps

1. Install `sphinx` and `sphinx_rtd_theme`:
   ```bash
   pip install sphinx sphinx_rtd_theme
   ```
2. Create a `docs` directory in the root of the project:
   ```bash
    mkdir docs
    ```
3. Create the documentation structure:
    ```bash
    cd docs
    sphinx-quickstart
    ```
    - Answer the questions as follows:
      - Separate source and build directories (y/n) [n]: n
      - Project name: PROTAC-Degradation-Predictor
      - Author name(s): Your Name
      - Project version: v1.0.1
    ```
4. Edit the `conf.py` file:
    - Add the following lines:
      ```python
      import os
      import sys
      sys.path.insert(0, os.path.abspath('..'))
      ```
    - Add the following line to the `extensions` list:
      ```python
      'sphinx.ext.autodoc',
      ```
    - Check the current `conf.py` file in this repository for more details.
5. Create modules rst files: `sphinx-apidoc -o docs/source/ ./protac_degradation_predictor`
6. To include modules in the documentation, in `index.rst`, add the following line: `source/modules`
7. Setup the gh-pages branch:
    ```bash
    git checkout --orphan gh-pages
    git reset --hard
    git commit --allow-empty -m "Init"
    git push origin gh-pages
    git checkout main
    ```
8. Generate the specific workflow Action file:
    - See file `.github/workflows/gh-pages.yml` in this repository.
9. The start page is in the file `index.rst`.

## Miscellaneous

- Useful [guide](https://olgarithms.github.io/sphinx-tutorial/docs/7-hosting-on-github-pages.html)
- Automatically create modules rst files: `sphinx-apidoc -o source/ ../protac_degradation_predictor`
  - Then, in `index.rst`, add the following line: `source/modules`
- Build the documentation: `make clean ; make html`