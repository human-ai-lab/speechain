"""Generate the code reference pages and navigation."""

# Import required modules
from pathlib import Path
import mkdocs_gen_files

# Initialize navigation object
nav = mkdocs_gen_files.Nav()

# Get root and source directories
root = Path(__file__).parent.parent
src = root / "speechain"

# exclude modules from being included in the documentation
exclude_modules = {str(src / "model/abs.py")}
# exclude_modules = {} 

# Iterate through all Python files in source directory
for path in sorted(src.rglob("*.py")):
    # exclude __pycache__ directories
    if "__pycache__" in path.parts or str(path) in exclude_modules:
        continue
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Add this check before setting nav
    if parts:  # Only add to nav if parts is not empty
        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Generate navigation summary file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  
    nav_file.writelines(nav.build_literate_nav())