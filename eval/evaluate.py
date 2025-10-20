import argparse, json, os
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="eval/prompts_example.jsonl")
    ap.add_argument("--api", type=str, default="http://localhost:8000/ask")
    args = ap.parse_args()

    os.makedirs("eval/logs", exist_ok=True)
    with open(args.file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            r = requests.post(args.api, json={"query": ex["prompt"], "top_k": ex.get("k", 4)}, timeout=60)
            out = r.json()
            with open(f"eval/logs/out_{i:03d}.json", "w", encoding="utf-8") as fo:
                json.dump(out, fo, ensure_ascii=False, indent=2)
            print(i, out.get("decision"))

if __name__ == "__main__":
    main()
