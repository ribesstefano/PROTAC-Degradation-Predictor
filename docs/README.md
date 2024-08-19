# Useful Tips on How to Build the Documentation

This document provides useful tips on how to build the documentation for the project.

- Automatically create modules rst files: `sphinx-apidoc -o source/ ../protac_degradation_predictor`
  - Then, in `index.rst`, add the following line: `source/modules`
- Build the documentation: `make clean ; make html`