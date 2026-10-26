"""Execute the notebook from a fresh kernel; exits nonzero on any cell error."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NOTEBOOK_NAME = "notebook.ipynb"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--write", action="store_true", help="Save executed cells to the notebook")
    args = parser.parse_args()
    path = ROOT / NOTEBOOK_NAME
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=180, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    if args.write:
        notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nbformat.write(notebook, path)
    print(f"Executed {path.name}: all cells passed")
