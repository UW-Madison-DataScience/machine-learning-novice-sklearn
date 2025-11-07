import os
import nbformat
import re
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Paths
episodes_dir = "episodes"
notebooks_dir = "notebooks"

# Base URL for images in the notebooks.
# Adjust this if your repo or branch name changes.
IMAGE_BASE_URL = (
    "https://raw.githubusercontent.com/UW-Madison-DataScience/"
    "machine-learning-novice-sklearn-v2/main/episodes/"
)

# List of Markdown files to ignore (no conversion needed)
ignore_list = [
    # "01-introduction.md",
]

# Ensure notebooks directory exists
os.makedirs(notebooks_dir, exist_ok=True)

# Regular expression to detect code blocks (matches ```language\n...\n```).
code_block_pattern = re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL)

# Regular expression to detect image links with local fig paths, e.g.:
# ![caption](fig/introduction/image.png)
# ![caption](../fig/introduction/image.png){alt='something'}
image_pattern = re.compile(
    r"!\[([^\]]*)\]\("       # ![alttext](
    r"(\.\./)?fig/([^)]+)"   # optional ../ then fig/filename
    r"\)(\{[^}]*\})?",       # optional {alt='...'} or other attrs
    re.MULTILINE,
)


def rewrite_image_paths(md_content: str) -> str:
    """
    Rewrite local image links that point to fig/... so they use a
    GitHub raw URL, which will render correctly inside notebooks.
    """

    def repl(match: re.Match) -> str:
        alt_text = match.group(1)
        filename = match.group(3)  # everything after fig/
        # We intentionally drop the Pandoc attribute block (group 4),
        # since Jupyter's markdown renderer doesn't use it.
        return f"![{alt_text}]({IMAGE_BASE_URL}fig/{filename})"

    return image_pattern.sub(repl, md_content)


def split_markdown(md_content: str):
    """
    Splits Markdown content into separate Markdown and Code cells
    based on fenced code blocks.
    """
    cells = []
    position = 0

    for match in code_block_pattern.finditer(md_content):
        # Text before the code block => Markdown cell
        before_code = md_content[position:match.start()].strip()
        if before_code:
            cells.append(new_markdown_cell(before_code))

        # Code block content => Code cell
        code_content = match.group(2).strip()
        if code_content:
            cells.append(new_code_cell(code_content))

        position = match.end()

    # Any remaining Markdown after the last code block
    remaining_md = md_content[position:].strip()
    if remaining_md:
        cells.append(new_markdown_cell(remaining_md))

    return cells


# Convert each Markdown file in episodes/
for filename in os.listdir(episodes_dir):
    if filename.endswith(".md") and filename not in ignore_list:
        md_path = os.path.join(episodes_dir, filename)
        ipynb_path = os.path.join(
            notebooks_dir, filename.replace(".md", ".ipynb")
        )

        # Read Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Rewrite image paths for notebook rendering
        md_content = rewrite_image_paths(md_content)

        # Split into Markdown and Code cells
        notebook_cells = split_markdown(md_content)

        # Create Jupyter notebook
        nb = new_notebook(cells=notebook_cells)

        # Save as .ipynb
        with open(ipynb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

print("Conversion complete! Excluded:", ", ".join(ignore_list))
