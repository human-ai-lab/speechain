## How to build speechain documentation

```bash
# install necessary packages
pip install mkdocs-material mkdocstrings[python]

# build documentation
mkdocs build -v

# serve documentation
mkdocs serve -v

# deploy to github pages, not mandatory auto build when push
mkdocs gh-deploy
```
