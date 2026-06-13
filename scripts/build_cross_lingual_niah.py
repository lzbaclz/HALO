"""Build a cross-lingual NIAH dataset for HALO Path D cross-language verification.

Each example:
  - Haystack: 8K tokens of zh + en filler text (proportional mixing)
  - Needle: "The special magic number is N123456." (en) or 中文版本
  - Question: Either zh or en
  - Answer: the number (language-agnostic)

We test 3 sub-configs:
  - zh-haystack + zh-question
  - en-haystack + en-question  (control, should match RULER NIAH baseline)
  - mixed (zh-haystack + en-question, the hardest case)

Outputs:
  experiments/cell_xlingual_niah/{zh,en,mixed}/data.jsonl
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path

ZH_FILLER = """春天来了，万物复苏，山上的花儿盛开了。农民们开始耕种，鸟儿在树梢上歌唱。
小河里的水变得清澈，鱼儿在水中游来游去。孩子们在草地上奔跑，放风筝。
夜晚的星空很美，月亮挂在天空上。城市的灯光闪烁，街道上人来人往。
学校里学生们认真学习，老师们辛勤地教导。图书馆里安静极了，只有翻书的声音。
工厂里机器在运转，工人们专心工作。医院里医生在治疗病人，护士们忙碌着。
餐厅里飘出了诱人的香味，厨师们正在准备晚餐。公园里老人们悠闲地散步，享受美好的时光。
火车站人山人海，旅客们提着行李等待发车。机场的飞机起飞和降落，连接着远方的城市。
"""

EN_FILLER = """The grass is green. The sky is blue. The sun is yellow. Here we go.
Trees are tall. Rivers flow. Mountains stand. Clouds drift. Wind blows softly.
Birds sing in the morning. Children play in the afternoon. Stars shine at night.
The forest is full of life. Animals run, hide, and forage. Flowers bloom each spring.
Cities buzz with activity. Roads carry cars and trucks. People work and rest in rhythm.
Schools teach students. Books hold knowledge. Libraries are quiet sanctuaries.
The ocean is deep and wide. Waves crash on the shore. Boats sail across the water.
Music fills the air at festivals. Food brings families together. Travel widens horizons.
"""

ZH_NEEDLE_TEMPLATE = "特殊的魔法数字是{number}。请记住它，稍后我会问你。"
EN_NEEDLE_TEMPLATE = "The special magic number is {number}. Remember it; I will quiz you later."

ZH_QUESTION = "请问特殊的魔法数字是什么？"
EN_QUESTION = "What is the special magic number?"


def repeat_to_length(filler: str, target_chars: int) -> str:
    out = []
    cur = 0
    while cur < target_chars:
        out.append(filler)
        cur += len(filler)
    return "".join(out)[:target_chars]


def build_example(haystack: str, question: str, needle_template: str,
                   answer_number: str, position_frac: float = 0.5) -> dict:
    needle = needle_template.format(number=answer_number)
    n = len(haystack)
    pos = int(position_frac * n)
    # Insert needle at position
    text_with_needle = haystack[:pos] + " " + needle + " " + haystack[pos:]
    return {
        "input": text_with_needle,
        "question": question,
        "answer": answer_number,
    }


def main():
    rng = random.Random(0)
    n_examples = 30
    target_chars = 3000 * 4  # ~6K tokens haystack  # rough estimate: 8K tokens ≈ 32K chars for mixed zh/en

    out_root = Path("experiments/cell_xlingual_niah")
    out_root.mkdir(parents=True, exist_ok=True)

    configs = {
        "zh": (ZH_FILLER, ZH_QUESTION, ZH_NEEDLE_TEMPLATE),
        "en": (EN_FILLER, EN_QUESTION, EN_NEEDLE_TEMPLATE),
        "mixed_zh_en": (ZH_FILLER + EN_FILLER, EN_QUESTION, EN_NEEDLE_TEMPLATE),
        "mixed_en_zh": (ZH_FILLER + EN_FILLER, ZH_QUESTION, ZH_NEEDLE_TEMPLATE),
    }

    for tag, (filler, question, needle_tpl) in configs.items():
        out_dir = out_root / tag
        out_dir.mkdir(exist_ok=True)
        rng_loc = random.Random(0 + hash(tag) % 1000)
        with open(out_dir / "data.jsonl", "w") as f:
            for i in range(n_examples):
                haystack = repeat_to_length(filler, target_chars)
                # Random 6-digit number
                number = f"N{rng_loc.randrange(100000, 999999)}"
                pos_frac = (i % 5 + 1) / 6.0  # spread positions 1/6 .. 5/6
                ex = build_example(haystack, question, needle_tpl, number, pos_frac)
                ex["index"] = i
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"wrote {out_dir}/data.jsonl ({n_examples} examples)")


if __name__ == "__main__":
    main()
