import argparse
from src.retriever import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to documents (txt)")
    args = ap.parse_args()
    build_index(args.path)
