"""Check that every number an analyst memo cites traces back to its payload. Writes out/memos/factcheck.json."""
import json
import re

from analyst import MEMOS

NUM = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")  # greedy, so "77.7th" stays 77.7
SKIP = re.compile(r"Bank \d+|T-\d|^\s*\d+\.\s", re.M)  # labels, quarter tags, list numbering


def candidates(payload):
    out = []
    for q in payload:
        out.append(q["risk_rank_pct"])
        for k in ("credit_lens", "run_lens", "mtm_cap", "run_gap"):
            if q.get(k) is not None:
                out += [q[k], q[k] * 100]
        for section in ("values", "peer_percentile", "score_contribution"):
            for v in q[section].values():
                if v is not None:
                    out += [v, v * 100]  # fractions are usually quoted as percentages
    return out


def traced(x, cands, whole):
    # a whole number ("about 13%") may round loosely; decimals must match closely
    tol = lambda c: 1.0 if whole else max(0.15, 0.02 * abs(c))
    return any(abs(abs(x) - abs(c)) <= tol(c) for c in cands)


def check(text, payload):
    cands = candidates(payload)
    text = SKIP.sub(" ", text)
    found = [m.group() for m in NUM.finditer(text)]
    nums = [(s, float(s.replace(",", "").replace("−", "-"))) for s in found]
    # small integers (e.g. "2-3 questions", "three quarters") are prose, not data
    nums = [(s, x) for s, x in nums if not (x.is_integer() and abs(x) <= 3)]
    bad = [s for s, x in nums if not traced(x, cands, "." not in s)]
    return {"cited": len(nums), "untraced": bad}


def main():
    report = {}
    for raw in sorted((MEMOS / "raw").glob("bank_*.md")):
        payload = json.loads((MEMOS / "payloads" / f"{raw.stem}.json").read_text())
        report[raw.stem] = check(raw.read_text(), payload)
    (MEMOS / "factcheck.json").write_text(json.dumps(report, indent=1))
    total = sum(r["cited"] for r in report.values())
    bad = sum(len(r["untraced"]) for r in report.values())
    print(f"{total} numbers cited, {bad} untraced ({bad / total:.1%})")
    for k, r in report.items():
        if r["untraced"]:
            print(k, r["untraced"])


if __name__ == "__main__":
    main()
