#!/usr/bin/env python3
"""Discourse-bench v2 — scaled-up generator (n=50+ per subtask).

Extends v1's template count to support claims that "Path D matches Full and
KIVI fails" with statistical power:

  v1: 12 prompts per subtask × 3 = 36 total; 2-3 templates per subtask
  v2: 50+ prompts per subtask × 3 = 150+ total; 8-10 templates per subtask
      across 3 difficulty bands (early / middle / late antecedent distance)

Subtasks (unchanged from v1):
  DA  – Long-distance pronominal anaphora
  BR  – Bridging causal inference
  CM  – Implicit commitment tracking

Subtask additions (v2):
  DA: 6 new templates (gendered + ungendered pronouns; possessives;
       multi-character disambiguation requiring last-mention recency)
  BR: 4 new templates (multi-hop bridging: §A states rule, §B states fact,
       §C states observation requiring A∧B)
  CM: 4 new templates (deadline / quota / proclamation / writ of
       prohibition; mixed yes/no expected answers)

Evidence-position bands controlled by --bands (default early,middle,late).
"early"/"middle"/"late" refer to the *position of the evidence (antecedent
or constraint) within the context window*, NOT to retrieval difficulty.
We deliberately avoid "near"/"far" because they invite the opposite
intuition depending on whether you mean "near the question" or "near the
context start":
  early:  evidence in opening ~5% of context (longest evidence→question distance)
  middle: evidence in middle ~50% of context
  late:   evidence in final ~5% of context (shortest evidence→question distance)
Recency-based commitment policies (StreamingLLM) are predicted to do best
on "late" (recent positions kept) and worst on "early"; query-aware
retrievers and identity-preserving policies should be band-invariant.

Determinism: seed-controlled. Re-running with same seed reproduces the JSONL
byte-for-byte.

This is still PILOT data — no human gold labels; gold strings are derived
from the template. Reviewers can verify templates are not adversarially tuned
by reading this source.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Reuse v1's filler primitives to keep the wiki-style distractor distribution
# identical across v1/v2 (so reviewers can pin any v1↔v2 score delta on the
# discourse anchor itself, not on filler text).
import importlib.util
import sys

_HERE = Path(__file__).resolve().parent
_V1_SPEC = importlib.util.spec_from_file_location("discourse_bench_v1", _HERE / "generate.py")
_V1 = importlib.util.module_from_spec(_V1_SPEC)
sys.modules["discourse_bench_v1"] = _V1
_V1_SPEC.loader.exec_module(_V1)

make_filler_block = _V1.make_filler_block


# --------------------------- DA: extended templates ---------------------------
DA_NAMES_FEM = _V1.DA_NAMES_FEM + [
    "Yui Hashimoto", "Anya Volkova", "Olamide Bankole", "Solange Marchetti",
    "Eshrat Saberi", "Lucia Restrepo", "Tassana Phichai", "Pavlina Stoyanova",
]
DA_NAMES_MASC = _V1.DA_NAMES_MASC + [
    "Hideki Tanigawa", "Bohdan Kovalenko", "Adeyemi Adebola", "Cosimo de'Salvi",
    "Farhad Naderi", "Augusto Bocanegra", "Wisarut Phromtep", "Kiril Dimitrov",
]
DA_NAMES_NEUT = [
    "Pat Sinclair", "Tay Whittle", "Rey Quintero", "Sam Ardent",
    "Dani Velthuis", "Jordan Maeso", "Robin Karagiannis", "Alex Olusola",
]

DA_TEMPLATES_V2 = list(_V1.DA_TEMPLATES) + [
    {
        "antecedent_sent": "{name} stored the {thing} inside the {place} before noon.",
        "pronoun_sent": "By dusk it was no longer there: someone had moved it.",
        "question": "Who placed the {thing} in the {place} before noon?",
        "actions": ["stored"],
        "things": ["coin", "ledger", "key", "necklace", "letter", "ring", "scroll"],
        "places": ["abbey", "guildhall", "registry", "vault", "scriptorium"],
        "pronoun_kind": "none",
    },
    {
        "antecedent_sent": "It was {name} who first reported the {event} to the council.",
        "pronoun_sent": "They were later commended for the disclosure.",
        "question": "Who first reported the {event} to the council?",
        "actions": [],
        "things": [],
        "places": [],
        "events": ["incident", "fire", "dispute", "transaction", "breach"],
        "events_for_q": True,
        "pronoun_kind": "neut",
    },
    {
        "antecedent_sent": "The {place} where the council met that morning was unlocked by {name}.",
        "pronoun_sent": "Her key was later found resting on the table.",
        "question": "Whose key was found on the table?",
        "actions": [],
        "things": [],
        "places": ["abbey", "guildhall", "registry", "treasury", "vestry"],
        "pronoun_kind": "fem",
    },
    {
        "antecedent_sent": "{name} carried the {thing} across the courtyard.",
        "pronoun_sent": "His shadow was the last anyone saw of him that day.",
        "question": "Who carried the {thing} across the courtyard?",
        "actions": [],
        "things": ["barrel", "satchel", "lantern", "chest", "bundle"],
        "places": [],
        "pronoun_kind": "masc",
    },
    {
        "antecedent_sent": "Among the apprentices, {name} alone refused the {thing}.",
        "pronoun_sent": "Their refusal was recorded by the clerk that evening.",
        "question": "Who alone refused the {thing}?",
        "actions": [],
        "things": ["pledge", "oath", "stipend", "writ", "appointment"],
        "places": [],
        "pronoun_kind": "neut",
    },
    {
        "antecedent_sent": "Long before the others arrived, {name} unlocked the {place}.",
        "pronoun_sent": "She had been holding the only spare key.",
        "question": "Who unlocked the {place} before the others arrived?",
        "actions": [],
        "things": [],
        "places": ["chamber", "scriptorium", "armoury", "library", "chapel"],
        "pronoun_kind": "fem",
    },
]


def _pick_name(rng: random.Random, kind: str) -> str:
    if kind == "fem":
        return rng.choice(DA_NAMES_FEM)
    elif kind == "masc":
        return rng.choice(DA_NAMES_MASC)
    elif kind == "neut":
        return rng.choice(DA_NAMES_NEUT + DA_NAMES_FEM + DA_NAMES_MASC)
    elif kind == "none":
        # no pronoun in the back-reference; gender irrelevant
        return rng.choice(DA_NAMES_FEM + DA_NAMES_MASC + DA_NAMES_NEUT)
    # legacy: she-in-pronoun-sent heuristic
    return rng.choice(DA_NAMES_FEM + DA_NAMES_MASC)


def make_da_prompt_v2(rng: random.Random, n_filler_paras: int, band: str) -> dict:
    """Extended DA prompt; band ∈ {early,middle,late} controls antecedent position."""
    tpl = rng.choice(DA_TEMPLATES_V2)
    kind = tpl.get("pronoun_kind", None)
    if kind is None:
        # legacy v1 templates: use heuristic
        is_fem = "she" in tpl["pronoun_sent"]
        name = rng.choice(DA_NAMES_FEM if is_fem else DA_NAMES_MASC)
    else:
        name = _pick_name(rng, kind)

    action = rng.choice(tpl["actions"]) if tpl["actions"] else ""
    thing = rng.choice(tpl["things"]) if tpl["things"] else ""
    place = rng.choice(tpl["places"]) if tpl["places"] else ""
    event = rng.choice(tpl["events"]) if tpl.get("events") else ""

    antecedent = tpl["antecedent_sent"].format(name=name, place=place, event=event, thing=thing)
    pronoun_sent = tpl["pronoun_sent"].format(action=action, thing=thing, event=event)
    if tpl.get("events_for_q"):
        question = tpl["question"].format(action=action, event=event, thing=thing, place=place)
    else:
        question = tpl["question"].format(action=action, thing=thing, place=place, event=event)

    # Band: place evidence at early (front), middle, or late (just before the question)
    if band == "early":
        f_before = 1
        f_after = max(1, n_filler_paras - 1)
    elif band == "middle":
        f_before = n_filler_paras // 2
        f_after = n_filler_paras - f_before
    else:  # late
        f_before = max(1, n_filler_paras - 1)
        f_after = 1

    filler1 = make_filler_block(rng, f_before)
    filler2 = make_filler_block(rng, f_after)
    body = (f"\n{filler1}\n\n{antecedent}\n\n{filler2}\n\n{pronoun_sent}\n\n"
            f"Question: {question}\nAnswer with only the name.\nAnswer:")
    return {
        "subtask": "DA",
        "band": band,
        "prompt": body,
        "gold": [name, name.split()[0], name.split()[-1]],
        "antecedent": antecedent,
        "pronoun_sent": pronoun_sent,
        "question": question,
        "template_id": DA_TEMPLATES_V2.index(tpl),
    }


# --------------------------- BR: extended templates (with multi-hop) ---------------------------
BR_TEMPLATES_V2 = list(_V1.BR_TEMPLATES) + [
    {
        # Multi-hop: §A rule, §B fact, §C observation → why §C?
        "constraint": "Royal decree {decree_num} forbids any guild member to vote before completing {years} years of service.",
        "bridge_fact": "{name} was admitted to the guild in {start_year}.",
        "observation": "At the {target_year} council assembly, {name}'s vote was rejected by the clerk.",
        "question": "Why was {name}'s vote at the {target_year} assembly rejected?",
        "gold_pattern": ["{years} year", "service", "not yet", "ineligible", "decree", "{decree_num}"],
        "n_hops": 2,
    },
    {
        "constraint": "The Treasury rule of {year_rule} declares that any silver shipment above {weight} marks requires a stamped manifest signed by two assayers.",
        "bridge_fact": "{name}'s consignment weighed {ship_weight} marks of silver.",
        "observation": "On arrival, the harbourmaster impounded {name}'s consignment.",
        "question": "Why did the harbourmaster impound {name}'s consignment?",
        "gold_pattern": ["manifest", "two assayers", "Treasury rule", "stamped", "without"],
        "n_hops": 2,
    },
    {
        "constraint": "Provincial ordinance §{section} stipulates that wells dug deeper than {depth} cubits within town walls must be inspected within {grace} days.",
        "observation": "Inspectors arrived at {name}'s well only after {actual} days, finding it sealed.",
        "question": "Why might the inspectors have found {name}'s well sealed?",
        "gold_pattern": ["{grace} day", "inspection", "overdue", "exceeded", "ordinance"],
        "n_hops": 1,
    },
    {
        "constraint": "Hanseatic League charter article {art} bans the resale of unstamped cloth at any League fair.",
        "bridge_fact": "{name}'s consignment of cloth bore no stamp.",
        "observation": "The fair-stewards confiscated {name}'s consignment without compensation.",
        "question": "Why was {name}'s consignment confiscated?",
        "gold_pattern": ["unstamped", "ban", "charter", "{art}", "article", "resale"],
        "n_hops": 2,
    },
]

BR_NAMES = DA_NAMES_FEM + DA_NAMES_MASC + DA_NAMES_NEUT


def make_br_prompt_v2(rng: random.Random, n_filler_paras: int, band: str) -> dict:
    tpl = rng.choice(BR_TEMPLATES_V2)
    name = rng.choice(BR_NAMES)
    fmt = {"name": name}
    # Common fillers
    if "{section}" in tpl["constraint"]:
        fmt["section"] = str(rng.randint(10, 90))
    if "{cap}" in tpl["constraint"]:
        cap = rng.randint(20, 60); fmt["cap"] = str(cap)
        fmt["bag_count"] = str(rng.randint(5, 12))
        fmt["sack_size"] = str(rng.randint(6, 15))
        fmt["actual"] = str(cap)
    if "{pct}" in tpl["constraint"]:
        pct = rng.choice([5, 8, 10, 12, 15]); fmt["pct"] = str(pct)
        gross = rng.randint(100, 300); fmt["gross"] = str(gross)
        fmt["net"] = str(int(gross * (100 - pct) / 100))
    if "{start_year}" in tpl.get("observation", "") + tpl.get("bridge_fact", "") + tpl["constraint"]:
        sy = rng.randint(1450, 1550); fmt["start_year"] = str(sy)
        fmt["actual_year"] = str(sy + rng.randint(7, 10))
    # v2 multi-hop fillers
    if "{decree_num}" in tpl["constraint"]:
        fmt["decree_num"] = f"§{rng.randint(100, 999)}"
    if "{years}" in tpl["constraint"]:
        years = rng.choice([5, 7, 10, 12])
        fmt["years"] = str(years)
        sy = rng.randint(1450, 1500); fmt["start_year"] = str(sy)
        # target year before completion
        fmt["target_year"] = str(sy + rng.randint(1, years - 1))
    if "{year_rule}" in tpl["constraint"]:
        fmt["year_rule"] = str(rng.randint(1400, 1550))
        w = rng.randint(50, 150); fmt["weight"] = str(w)
        fmt["ship_weight"] = str(w + rng.randint(20, 80))
    if "{depth}" in tpl["constraint"]:
        fmt["depth"] = str(rng.randint(4, 12))
        grace = rng.randint(7, 30); fmt["grace"] = str(grace)
        fmt["actual"] = str(grace + rng.randint(5, 20))
    if "{art}" in tpl["constraint"]:
        fmt["art"] = f"§{rng.randint(1, 200)}"

    constraint = tpl["constraint"].format(**fmt)
    bridge_fact = tpl.get("bridge_fact", "").format(**fmt) if tpl.get("bridge_fact") else ""
    observation = tpl["observation"].format(**fmt)
    question = tpl["question"].format(**fmt)
    gold = [g.format(**fmt) for g in tpl["gold_pattern"]]

    # Band: place constraint at front (early), middle, or close to question (late).
    if band == "early":
        gap_before, gap_mid, gap_after = max(1, n_filler_paras - 2), 1, 1
    elif band == "middle":
        gap_before = n_filler_paras // 3
        gap_mid = n_filler_paras // 3
        gap_after = n_filler_paras - gap_before - gap_mid
    else:  # late
        gap_before, gap_mid, gap_after = 1, 1, max(1, n_filler_paras - 2)

    f1 = make_filler_block(rng, gap_before)
    f2 = make_filler_block(rng, gap_mid)
    f3 = make_filler_block(rng, gap_after)
    bridge_block = (bridge_fact + "\n\n") if bridge_fact else ""
    body = (f"\n{f1}\n\n{constraint}\n\n{f2}\n\n{bridge_block}{observation}\n\n{f3}\n\n"
            f"Question: {question}\nAnswer concisely.\nAnswer:")
    return {
        "subtask": "BR",
        "band": band,
        "prompt": body,
        "gold": gold,
        "constraint": constraint,
        "bridge_fact": bridge_fact,
        "observation": observation,
        "question": question,
        "n_hops": tpl.get("n_hops", 1),
        "template_id": BR_TEMPLATES_V2.index(tpl),
    }


# --------------------------- CM: extended templates ---------------------------
CM_TEMPLATES_V2 = list(_V1.CM_TEMPLATES) + [
    {
        "commitment": "The municipal quota allows each household at most {cap} bushels of grain per quarter.",
        "query_yes_about_no": False,  # ask about a NO situation
        "query": "Would a household acquiring {extra} bushels in a single quarter be in compliance?",
        "extra_eval": lambda fmt: int(fmt["cap"]) + 10,
        "answer_no": ["no", "not in compliance", "exceeds", "above", "violation", "not compliant", "no."],
        "answer_yes": ["yes"],
        "expect": "no",
    },
    {
        "commitment": "The writ of prohibition forbids any trade in {good} on the day of the festival of {feast}.",
        "query": "May {good} be traded on the festival of {feast} under the writ?",
        "answer_no": ["no", "forbidden", "prohibited", "not allowed", "not permitted", "no."],
        "answer_yes": ["yes", "permitted", "allowed"],
        "expect": "no",
    },
    {
        "commitment": "Article {art} of the agreement permits the {role} to issue exemptions up to a maximum of {cap} per month.",
        "query": "Could the {role} legitimately issue {actual} exemptions in a single month under the agreement?",
        "extra_eval": lambda fmt: int(fmt["cap"]) + 3,
        "answer_no": ["no", "exceeds", "above", "not allowed", "not permitted", "no."],
        "answer_yes": ["yes"],
        "expect": "no",
    },
    {
        # YES case (variety)
        "commitment": "The treaty stipulates that any party may withdraw with {notice} months' written notice.",
        "query": "If {name} gives {actual_notice} months' written notice of withdrawal, would the withdrawal be valid?",
        "extra_eval_yes": lambda fmt: int(fmt["notice"]) + 2,
        "answer_yes": ["yes", "valid", "permitted", "allowed", "in compliance", "yes."],
        "answer_no": ["no", "invalid"],
        "expect": "yes",
    },
]

CM_FEASTS = ["Saint Anselm", "the Ascension", "Saint Brendan", "the Epiphany",
             "Saint Vlasios", "Saint Demetrius", "the Conversion", "Saint Eulalia"]


def make_cm_prompt_v2(rng: random.Random, n_filler_paras: int, band: str) -> dict:
    tpl = rng.choice(CM_TEMPLATES_V2)
    good = rng.choice(_V1.CM_GOODS)
    place = rng.choice(_V1.CM_PLACES)
    deadline_month, deadline_day = rng.choice([("June", 15), ("September", 1), ("November", 30), ("April", 7)])
    test_month, test_day = rng.choice([("July", 1), ("October", 15), ("December", 20), ("May", 5)])
    name = rng.choice(BR_NAMES)
    cap = rng.randint(20, 80)
    art = f"§{rng.randint(1, 200)}"
    feast = rng.choice(CM_FEASTS)
    role = rng.choice(["consul", "magistrate", "intendant", "warden", "syndic"])
    notice = rng.randint(3, 12)

    fmt = {
        "good": good, "place": place,
        "deadline": f"{deadline_month} {deadline_day}",
        "test_date": f"{test_month} {test_day}",
        "cap": str(cap), "art": art, "feast": feast,
        "role": role, "name": name,
        "notice": str(notice),
    }
    # Resolve extra_eval
    if "extra_eval" in tpl:
        fmt["extra"] = str(tpl["extra_eval"](fmt))
        fmt["actual"] = str(tpl["extra_eval"](fmt))
    if "extra_eval_yes" in tpl:
        fmt["actual_notice"] = str(tpl["extra_eval_yes"](fmt))

    commitment = tpl["commitment"].format(**fmt)
    query = tpl["query"].format(**fmt)
    if tpl.get("expect", "no") == "yes":
        gold = tpl["answer_yes"]
    else:
        gold = tpl["answer_no"]

    if band == "early":
        f_before, f_after = max(1, n_filler_paras - 1), 1
    elif band == "middle":
        f_before = n_filler_paras // 2
        f_after = n_filler_paras - f_before
    else:
        f_before, f_after = 1, max(1, n_filler_paras - 1)

    filler1 = make_filler_block(rng, f_before)
    filler2 = make_filler_block(rng, f_after)
    body = (f"\n{filler1}\n\n{commitment}\n\n{filler2}\n\n"
            f"Question: {query}\nAnswer yes or no with a brief explanation.\nAnswer:")
    return {
        "subtask": "CM",
        "band": band,
        "prompt": body,
        "gold": gold,
        "commitment": commitment,
        "query": query,
        "expect": tpl.get("expect", "no"),
        "template_id": CM_TEMPLATES_V2.index(tpl),
    }


# --------------------------- Driver ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-subtask", type=int, default=50,
                    help="Total prompts per subtask (split evenly across bands).")
    ap.add_argument("--target-ctx-tokens", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bands", type=str, default="early,middle,late",
                    help="Comma-separated subset of {early,middle,late}.")
    ap.add_argument("--out", type=str,
                    default="experiments/discourse_benchmark/discourse_eval_v2.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    assert all(b in {"early", "middle", "late"} for b in bands), bands

    paras_needed = max(20, args.target_ctx_tokens // 100)
    per_band = max(1, (args.n_per_subtask + len(bands) - 1) // len(bands))  # ceil

    prompts: list[dict] = []
    for sub_make in (make_da_prompt_v2, make_br_prompt_v2, make_cm_prompt_v2):
        for band in bands:
            for _ in range(per_band):
                prompts.append(sub_make(rng, paras_needed, band))

    char_counts = [len(p["prompt"]) for p in prompts]
    print(f"Generated {len(prompts)} prompts (≈{per_band} per band per subtask, {len(bands)} bands).")
    print(f"  chars   min={min(char_counts)} median={sorted(char_counts)[len(char_counts)//2]} max={max(char_counts)}")
    print(f"  tokens (approx) min={min(char_counts)//4} median={sorted(char_counts)[len(char_counts)//2]//4} max={max(char_counts)//4}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, p in enumerate(prompts):
            row = {
                "index": i,
                "subtask": p["subtask"],
                "band": p["band"],
                "template_id": p["template_id"],
                "input": p["prompt"],
                "outputs": p["gold"],
            }
            if "expect" in p:
                row["expect"] = p["expect"]
            if "n_hops" in p:
                row["n_hops"] = p["n_hops"]
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
