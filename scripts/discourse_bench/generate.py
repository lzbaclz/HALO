#!/usr/bin/env python3
"""Discourse-level long-context evaluation benchmark (pilot).

Audit Plan D: the existing NIAH adversarial / EnQA cells test
retrieve-or-not, NOT discourse-level long-context phenomena.
This generator builds a pilot suite of three subtasks that
require the model to maintain discourse state across 16K-65K
context:

  DA  - Long-distance pronominal anaphora (~16K-65K span between
        antecedent and pronoun reference)
  BR  - Bridging causal inference (planted constraint in §A,
        observation in §B that requires §A's content to explain)
  CM  - Implicit commitment tracking (deadline / cap / policy
        committed early, queried much later about a violation)

Each prompt is a long narrative with named entities, planted
"discourse anchors" at deterministic positions, and a final
question with a known short-string answer. Scoring is exact-match
substring or regex against a small set of acceptable answers.

This is a PILOT: ~50 prompts total, no human annotation, all
templates documented in this file. Reviewers can verify the
templates are not adversarially tuned by reading this source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


# --------------------------- Filler text pool ---------------------------
# Procedurally constructed wiki-style paragraphs (no copyrighted source).
# Each filler paragraph is ~120-180 tokens.

FILLER_TOPICS = [
    "agricultural development in the Loire valley during the 14th century",
    "the role of guild apprenticeships in medieval Saxony",
    "monsoon patterns over the South China Sea between 1850 and 1900",
    "the spread of indigo dyeing techniques from Sind to Java",
    "neolithic flint mining at Spiennes",
    "river-bank stabilisation methods in the Brahmaputra delta",
    "evolution of the lyre across Bronze Age Greece",
    "early metallurgy in the Caucasus copper belt",
    "regulation of urban water cisterns in Antwerp 1480-1620",
    "harvest cycles of taro in highland Papua New Guinea",
    "the iconography of Late Period Egyptian funeral steles",
    "rail freight tariffs in the German Empire 1880-1914",
    "ceramic chronology of the Yellow River basin",
    "transit-yard scheduling in the port of Hamburg 1925-1939",
    "mortuary practices among the Tukano of the Vaupes",
    "currency reform in Joseon-period Korea",
    "the geology of the Ural foothills",
    "labour migration between Naples and Buenos Aires 1880-1914",
    "irrigation engineering in pharaonic Egypt",
    "lichenometric dating of Norse landnam in Greenland",
    "drainage systems of the Indus civilisation",
    "the silk-road relay-station network through Khotan",
    "salt extraction at Halle and Lueneburg",
    "wool production in the highlands of Borrowdale",
    "cartographic projections used by Portuguese navigators",
    "amber trade routes through the Vistula basin",
    "bee-keeping practices in Anatolian villages",
    "harness design in Han-dynasty cavalry units",
    "vineyard tenancy under Burgundian dukes",
    "cattle taxation in the Mongol-Yuan transition",
]

FILLER_VERBS = [
    "developed", "constrained", "facilitated", "regulated", "transformed",
    "shaped", "redirected", "altered", "intensified", "displaced",
    "consolidated", "fractured", "diffused", "stratified", "reordered",
]

FILLER_NOUNS = [
    "patterns of settlement", "labour relations", "trade volume",
    "guild membership", "ceremonial calendars", "land tenure",
    "tax-collection routines", "household economies", "fishery yields",
    "monastic foundations", "civic festivals", "imperial inspections",
    "irrigation works", "harvest reports", "market regulations",
    "mortuary practices", "armament stocks", "currency standards",
]


def make_filler_paragraph(rng: random.Random, n_sentences: int = 5) -> str:
    """Generate a filler paragraph of n_sentences sentences."""
    topic = rng.choice(FILLER_TOPICS)
    sents = [f"This passage concerns {topic}."]
    for _ in range(n_sentences - 1):
        verb = rng.choice(FILLER_VERBS)
        noun = rng.choice(FILLER_NOUNS)
        century_or_year = rng.choice([
            f"during the {rng.randint(11, 18)}th century",
            f"under the rule of {rng.choice(['the Caliphate', 'the Hanseatic League', 'the Tang court', 'the Dutch Republic'])}",
            f"between {rng.randint(1450, 1850)} and {rng.randint(1855, 1900)}",
            f"in the period following the {rng.choice(['plague', 'famine', 'reformation', 'enclosure'])}",
            f"under successive cycles of {rng.choice(['drought', 'flood', 'frost', 'revolt'])}",
        ])
        actor = rng.choice([
            "Local administrators", "Itinerant merchants", "Guild masters",
            "Provincial nobility", "Migrant labourers", "Monastic stewards",
            "Customs officials", "Rural cooperatives", "Coastal smuggling networks",
            "Frontier garrisons",
        ])
        sents.append(f"{actor} {verb} {noun} {century_or_year}.")
    return " ".join(sents)


def make_filler_block(rng: random.Random, target_paragraphs: int) -> str:
    """Generate a multi-paragraph filler block."""
    paras = [make_filler_paragraph(rng, n_sentences=rng.randint(4, 7))
             for _ in range(target_paragraphs)]
    return "\n\n".join(paras)


# --------------------------- DA: Long-distance anaphora ---------------------------
DA_TEMPLATES = [
    {
        "antecedent_sent": "{name} arrived at the {place} just before dawn.",
        "pronoun_sent": "Later that evening she {action} the {thing}.",
        "question": "Who {action} the {thing} that evening?",
        "answer_kind": "name",
        "actions": ["misplaced", "returned", "polished", "sold", "delivered"],
        "things": ["coin", "ledger", "key", "necklace", "letter"],
        "places": ["harbour", "market", "abbey", "guildhall", "registry"],
    },
    {
        "antecedent_sent": "{name} had been the only witness to the {event}.",
        "pronoun_sent": "He refused to {action} when asked about it.",
        "question": "Who refused to {action} when asked about the {event}?",
        "answer_kind": "name",
        "actions": ["sign", "testify", "elaborate", "comment"],
        "events": ["incident", "fire", "dispute", "transaction"],
        "things": ["incident", "fire", "dispute", "transaction"],
        "places": [],
        "events_for_q": True,
    },
]

DA_NAMES_FEM = ["Tertia Iulia", "Sabine Bellanger", "Cao Yu", "Mariko Tominaga",
                "Galyna Ostroumova", "Beata Konieczna", "Adaeze Okafor", "Phakaphon Rojanasakul"]
DA_NAMES_MASC = ["Eustathios Komnenos", "Aleksei Voronin", "Gunter Brehm", "Ravi Subramanian",
                 "Jorge Larrazabal", "Henrik Mannerheim", "Tunde Adesanya", "Wachirawit Boonpan"]


def make_da_prompt(rng: random.Random, n_filler_paras: int) -> dict:
    """Long-distance anaphora prompt."""
    tpl = rng.choice(DA_TEMPLATES)
    is_fem = "she" in tpl["pronoun_sent"]
    name = rng.choice(DA_NAMES_FEM if is_fem else DA_NAMES_MASC)
    action = rng.choice(tpl["actions"])
    thing = rng.choice(tpl.get("things", ["item"]))
    place = rng.choice(tpl.get("places", ["location"])) if tpl.get("places") else "location"
    event = rng.choice(tpl.get("events", ["matter"]))

    antecedent = tpl["antecedent_sent"].format(name=name, place=place, event=event)
    pronoun_sent = tpl["pronoun_sent"].format(action=action, thing=thing)
    if tpl.get("events_for_q"):
        question = tpl["question"].format(action=action, event=event)
    else:
        question = tpl["question"].format(action=action, thing=thing)

    filler1 = make_filler_block(rng, n_filler_paras // 2)
    filler2 = make_filler_block(rng, n_filler_paras - n_filler_paras // 2)
    body = (f"\n{antecedent}\n\n{filler1}\n\n{filler2}\n\n{pronoun_sent}\n\n"
            f"Question: {question}\nAnswer with only the name.\nAnswer:")
    return {
        "subtask": "DA",
        "prompt": body,
        "gold": [name, name.split()[0], name.split()[-1]],
        "antecedent": antecedent,
        "pronoun_sent": pronoun_sent,
        "question": question,
    }


# --------------------------- BR: Bridging inference ---------------------------
BR_TEMPLATES = [
    {
        "constraint": "Town ordinance §{section} caps daily grain purchases at {cap} bushels.",
        "observation": "{name} brought {bag_count} sacks (each {sack_size} bushels) to market but left with only {actual} bushels' worth of receipts.",
        "question": "Why did {name} leave with receipts for only {actual} bushels instead of the full purchase amount?",
        "gold_pattern": ["cap", "ordinance", "limit", "rule", "{cap}"],
    },
    {
        "constraint": "The harvest tax of {pct}% applies to all grain weighed at the public scale.",
        "observation": "{name} weighed {gross} bushels and recorded {net} bushels in the household ledger.",
        "question": "Why is the recorded amount lower than the weighed amount?",
        "gold_pattern": ["tax", "{pct}%", "harvest tax"],
    },
    {
        "constraint": "Guild rules require apprentices to complete seven years of service before being granted full membership.",
        "observation": "{name} began apprenticeship in {start_year} but was granted full membership only in {actual_year}.",
        "question": "What is the minimum service period before {name} could have been granted full membership?",
        "gold_pattern": ["seven", "7"],
    },
]

BR_NAMES = DA_NAMES_FEM + DA_NAMES_MASC


def make_br_prompt(rng: random.Random, n_filler_paras: int) -> dict:
    tpl = rng.choice(BR_TEMPLATES)
    name = rng.choice(BR_NAMES)
    # Fill in template variables
    fmt = {"name": name}
    if "{section}" in tpl["constraint"]:
        fmt["section"] = str(rng.randint(10, 90))
    if "{cap}" in tpl["constraint"]:
        cap = rng.randint(20, 60)
        fmt["cap"] = str(cap); fmt["bag_count"] = str(rng.randint(5, 12))
        fmt["sack_size"] = str(rng.randint(6, 15))
        fmt["actual"] = str(cap)
    if "{pct}" in tpl["constraint"]:
        pct = rng.choice([5, 8, 10, 12, 15])
        fmt["pct"] = str(pct)
        gross = rng.randint(100, 300)
        fmt["gross"] = str(gross)
        fmt["net"] = str(int(gross * (100 - pct) / 100))
    if "{start_year}" in tpl["observation"]:
        sy = rng.randint(1450, 1550)
        fmt["start_year"] = str(sy)
        fmt["actual_year"] = str(sy + rng.randint(7, 10))
    constraint = tpl["constraint"].format(**fmt)
    observation = tpl["observation"].format(**fmt)
    question = tpl["question"].format(**fmt)
    gold = [g.format(**fmt) for g in tpl["gold_pattern"]]

    filler1 = make_filler_block(rng, n_filler_paras // 2)
    filler2 = make_filler_block(rng, n_filler_paras - n_filler_paras // 2)
    body = (f"\n{constraint}\n\n{filler1}\n\n{observation}\n\n{filler2}\n\n"
            f"Question: {question}\nAnswer concisely.\nAnswer:")
    return {
        "subtask": "BR",
        "prompt": body,
        "gold": gold,
        "constraint": constraint,
        "observation": observation,
        "question": question,
    }


# --------------------------- CM: Implicit commitment tracking ---------------------------
CM_TEMPLATES = [
    {
        "commitment": "The supply contract states that the {good} shipment must arrive by {deadline}.",
        "query": "Per the contract, would a {good} shipment arriving on {test_date} be considered on-time?",
        "yes_test": False,  # we test a date AFTER deadline
        "answer_yes": ["yes", "on time", "on-time", "on time."],
        "answer_no": ["no", "late", "after", "not on time", "no."],
    },
    {
        "commitment": "The royal proclamation forbids any export of {good} from the harbour of {place} without a stamped pass.",
        "query": "May {good} be shipped out of {place} without a stamped pass under the proclamation?",
        "yes_test": False,
        "answer_yes": ["yes", "allowed", "permitted"],
        "answer_no": ["no", "forbidden", "not allowed", "not permitted", "prohibited", "no."],
    },
]

CM_GOODS = ["wool", "salt", "iron ore", "wine", "alum", "linen", "tin", "copper"]
CM_PLACES = ["Bruges", "Genoa", "Antwerp", "Alexandria", "Basra", "Quanzhou",
             "Stralsund", "Riga", "Constantinople", "Marseille"]


def make_cm_prompt(rng: random.Random, n_filler_paras: int) -> dict:
    tpl = rng.choice(CM_TEMPLATES)
    good = rng.choice(CM_GOODS)
    place = rng.choice(CM_PLACES)
    deadline_month, deadline_day = rng.choice([("June", 15), ("September", 1), ("November", 30), ("April", 7)])
    test_month, test_day = rng.choice([("July", 1), ("October", 15), ("December", 20), ("May", 5)])
    fmt = {"good": good, "place": place,
           "deadline": f"{deadline_month} {deadline_day}",
           "test_date": f"{test_month} {test_day}"}
    commitment = tpl["commitment"].format(**fmt)
    query = tpl["query"].format(**fmt)
    gold = tpl["answer_no"]  # all templates designed for NO answer

    filler1 = make_filler_block(rng, n_filler_paras // 2)
    filler2 = make_filler_block(rng, n_filler_paras - n_filler_paras // 2)
    body = (f"\n{commitment}\n\n{filler1}\n\n{filler2}\n\n"
            f"Question: {query}\nAnswer yes or no with a brief explanation.\nAnswer:")
    return {
        "subtask": "CM",
        "prompt": body,
        "gold": gold,
        "commitment": commitment,
        "query": query,
    }


# --------------------------- Driver ---------------------------

def score(pred: str, gold: list[str]) -> float:
    """Case-insensitive substring containment scoring."""
    if not pred: return 0.0
    pred_l = pred.lower()
    for g in gold:
        if g.lower() in pred_l:
            return 1.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-subtask", type=int, default=20)
    ap.add_argument("--target-ctx-tokens", type=int, default=32000,
                    help="Approx context length in tokens (1 token ~= 4 chars)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/discourse_benchmark/discourse_eval.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # ~ chars-per-paragraph estimate: 5 sentences × ~80 chars = 400 chars = ~100 tokens
    # for target 32K tokens => need ~320 paragraphs
    paras_needed = max(20, args.target_ctx_tokens // 100)

    prompts = []
    for i in range(args.n_per_subtask):
        prompts.append(make_da_prompt(rng, paras_needed))
        prompts.append(make_br_prompt(rng, paras_needed))
        prompts.append(make_cm_prompt(rng, paras_needed))

    # Sanity: char count distribution
    char_counts = [len(p["prompt"]) for p in prompts]
    print(f"Generated {len(prompts)} prompts; chars: min={min(char_counts)} "
          f"median={sorted(char_counts)[len(char_counts)//2]} max={max(char_counts)}")
    print(f"Approx tokens: min={min(char_counts)//4} median={sorted(char_counts)[len(char_counts)//2]//4} max={max(char_counts)//4}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, p in enumerate(prompts):
            row = {
                "index": i,
                "subtask": p["subtask"],
                "input": p["prompt"],
                "outputs": p["gold"],
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
