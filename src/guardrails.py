import re
import yaml
from typing import Dict, Any, List, Tuple

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
except Exception:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None

class Guardrails:
    def __init__(self, policy_path: str = "policies/policy.yaml") -> None:
        with open(policy_path, "r", encoding="utf-8") as f:
            self.policy = yaml.safe_load(f)
        self.classifier = None
        self.tokenizer = None
        clf_cfg = self.policy.get("classifier", {})
        if clf_cfg.get("enabled") and AutoModelForSequenceClassification is not None:
            model_name = clf_cfg.get("model_name", "unitary/unbiased-toxic-roberta")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.classifier.eval()

    def _keyword_hits(self, text: str) -> List[Tuple[str, str]]:
        hits = []
        for cat, cfg in self.policy.get("categories", {}).items():
            for kw in cfg.get("keywords", []):
                if isinstance(kw, list):
                    kws = kw
                else:
                    kws = [kw]
                for k in kws:
                    if k and re.search(rf"\b{re.escape(k)}\b", text, flags=re.IGNORECASE):
                        hits.append((cat, k))
        return hits

    def _classifier_score(self, text: str) -> float:
        if not self.classifier or not self.tokenizer:
            return 0.0
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        # Heuristic: assume index 1 corresponds to 'toxic' for a binary classifier.
        toxic_prob = float(probs[-1])
        return toxic_prob

    def check(self, text: str) -> Dict[str, Any]:
        decision_log = {"keyword_hits": [], "classifier": {}, "decisions": []}
        hits = self._keyword_hits(text)
        decision_log["keyword_hits"] = hits

        toxic_prob = self._classifier_score(text) if hits or self.policy.get("classifier", {}).get("enabled") else 0.0
        decision_log["classifier"] = {"toxic_prob": toxic_prob}

        # Decide per category
        violations = []
        for cat, cfg in self.policy.get("categories", {}).items():
            thresh = float(cfg.get("threshold", 1.0))
            action = cfg.get("action", "block")
            cat_hit = any(h[0] == cat for h in hits) or (toxic_prob >= thresh)
            if cat_hit:
                violations.append({"category": cat, "action": action, "toxic_prob": toxic_prob})

        decision = "allow" if not violations else max(violations, key=lambda v: (v["action"] != "allow", v["toxic_prob"]))["action"]
        decision_log["decisions"] = violations
        return {"decision": decision, "log": decision_log}

    def safe_respond(self, user_text: str) -> str:
        return "I'm here to help, but I can’t assist with that request. If you have another question or want general information, let me know."
