# LLMs for Range Prediction of Electric Trucks in Pre-Development Software and Electronics

Still under active development.

# Project Documentation

This project uses Sphinx to generate documentation directly from the source code.
You can build the documentation locally by following the steps below or access the hosted version under this link (available until 30.04.2026): 

## Setup

Install the required dependencies:

pip install sphinx sphinx-autobuild
pip install sphinx-rtd-theme # optional (recommended)


## Generate API Documentation

From the project root directory run:


cd src
sphinx-apidoc -o ../docs/source . --force
cd ..

This scans the source code and generates .rst files inside docs/source/.

Build and Serve the Documentation

Start a live documentation server with automatic rebuilds:

sphinx-autobuild docs/source docs/build/html

Open your browser usually here: http://127.0.0.1:8000

The documentation reloads automatically when source code or documentation files change.

Static Build (Optional)

To generate a static HTML version without live reload:

sphinx-build docs/source docs/build/html

The generated documentation will be available at:

docs/build/html/index.html

# Note on the use of AI tools
In accordance with the guidelines of Technische Universität Darmstadt, AI tools were 
used to assist the work process and did not replace independent work. The code used
in this work was written independently and partially revised with the help of ChatGPT (OpenAI) and
Claude (Anthropic). The code parts revised in this way are marked with comments in the source code.
Docstrings follow the Google style and were written with assistance from Claude (Anthropic).


