#!/bin/bash

# This script generates the documentation for the project.

# Run this script from the root of the repository using:
# chmod +x docs/gen_docs.sh && ./docs/gen_docs.sh

# Install the documentation requirements
pip install -r requirements/docs.txt

# Generate the documentation
sphinx-build -b html docs docs/_build/html