from pathlib import Path


ROOT_DIR = Path(__file__).absolute().parent.parent
MODELS_DIR = ROOT_DIR / "models"
GRAPH_FILE = MODELS_DIR / "graph.pkl"
DISTANCES_FILE = MODELS_DIR / "distances.pkl"


if __name__ == '__main__':
    raise SystemExit("Cannot run this file.")
