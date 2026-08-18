"""Generate the FROZEN 300-item static eval. Run once at week 0, never again.

Design constraints:
- Fully deterministic (seed 42): the entire set reproduces from this script.
- Every item is machine-checkable (exact or numeric) — no judge in the public
  curves, so scoring is beyond dispute.
- 6 skills x 50 items. Difficulty mixed within each skill.
- The eval harness lowercases exact matches, so no case-sensitive tasks.

After generation, record the sha256 in config/protocol.yaml. Any future
change to eval/static/static_eval.jsonl invalidates the experiment.

    python scripts/build_static_eval.py
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

rng = random.Random(42)
SEEN: set[str] = set()  # global prompt dedupe — retry until unique
OUT = Path(__file__).resolve().parents[1] / "eval" / "static" / "static_eval.jsonl"

NAMES = ["Asha", "Ben", "Chen", "Diya", "Emil", "Fatima", "Goro", "Hana", "Ivan", "Jun",
         "Kira", "Leo", "Mina", "Noor", "Omar", "Priya", "Quinn", "Rosa", "Sam", "Tara"]
ITEMS = ["apples", "notebooks", "tickets", "marbles", "stickers", "coins", "bricks", "pens"]


def make_math(n: int) -> list[dict]:
    out = []
    i = 0
    while i < n:
        kind = i % 5
        a, b, c = rng.randint(3, 40), rng.randint(2, 12), rng.randint(2, 9)
        name, name2 = rng.sample(NAMES, 2)
        item = rng.choice(ITEMS)
        if kind == 0:  # buy/give
            ans = a * b - c
            p = (f"{name} buys {a} packs of {item} with {b} in each pack, "
                 f"then gives away {c}. How many {item} does {name} have left?")
        elif kind == 1:  # unit price
            price, qty = rng.randint(2, 15), rng.randint(3, 20)
            ans = price * qty
            p = f"One {item[:-1]} costs {price} dollars. How much do {qty} cost, in dollars?"
        elif kind == 2:  # split remainder
            total = b * c * 2 + rng.randint(0, b - 1)
            ans = total % b
            p = (f"{name} and {name2} share a box of {total} {item} by dealing them out "
                 f"equally to {b} people. How many {item} are left over?")
        elif kind == 3:  # rate * time
            rate, hours = rng.randint(4, 60), rng.randint(2, 12)
            ans = rate * hours
            p = f"A machine produces {rate} {item} per hour. How many does it produce in {hours} hours?"
        else:  # two-step percent-free average
            x, y, z = rng.randint(10, 50), rng.randint(10, 50), rng.randint(10, 50)
            while (x + y + z) % 3:
                z += 1
            ans = (x + y + z) // 3
            p = f"{name} scores {x}, {y}, and {z} on three quizzes. What is the average score?"
        if p in SEEN:
            continue
        SEEN.add(p)
        i += 1
        out.append({"id": f"math-{i:03d}", "skill": "math", "prompt": p + " Answer with a number.",
                    "check": "numeric", "answer": str(ans)})
    return out


def make_reasoning(n: int) -> list[dict]:
    out = []
    i = 0
    while i < n:
        kind = i % 5
        if kind == 0:  # transitive ordering
            a, b, c = rng.sample(NAMES, 3)
            p = (f"{a} is taller than {b}. {b} is taller than {c}. "
                 f"Who is the shortest? Answer with just the name.")
            ans = c
        elif kind == 1:  # weekday arithmetic
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            start, delta = rng.randrange(7), rng.randint(3, 30)
            p = (f"Today is {days[start]}. What day of the week is it {delta} days from now? "
                 f"Answer with just the day name.")
            ans = days[(start + delta) % 7]
        elif kind == 2:  # arithmetic sequence
            first, step = rng.randint(1, 9), rng.randint(2, 7)
            k = rng.randint(5, 12)
            seq = [first + j * step for j in range(4)]
            ans = str(first + (k - 1) * step)
            p = (f"The sequence {', '.join(map(str, seq))}, ... continues with the same rule. "
                 f"What is term number {k}? Answer with a number.")
        elif kind == 3:  # inverted condition counting
            total = rng.randint(12, 30)
            red = rng.randint(3, total - 3)
            p = (f"A bag has {total} marbles; {red} are red and the rest are blue. "
                 f"How many are blue? Answer with a number.")
            ans = str(total - red)
        else:  # truth-teller lite / negation
            a, b = rng.sample(NAMES, 2)
            item = rng.choice(ITEMS)
            x = rng.randint(2, 20)
            p = (f"{a} says: \"{b} has fewer than {x} {item}.\" "
                 f"{b} actually has {x + rng.randint(1, 5)} {item}. "
                 f"Is {a}'s statement true or false? Answer true or false.")
            ans = "false"
        if p in SEEN:
            continue
        SEEN.add(p)
        i += 1
        out.append({"id": f"reas-{i:03d}", "skill": "reasoning", "prompt": p,
                    "check": "numeric" if str(ans).isdigit() else "exact", "answer": str(ans)})
    return out


def make_instruction(n: int) -> list[dict]:
    words = ["orbit", "velvet", "quartz", "meadow", "signal", "harbor", "cinder", "pluto",
             "lattice", "ember"]
    out = []
    i = 0
    while i < n:
        kind = i % 5
        if kind == 0:
            w, k = rng.choice(words), rng.randint(2, 5)
            p = f"Repeat the word '{w}' exactly {k} times, separated by single hyphens, with nothing else."
            ans = "-".join([w] * k)
        elif kind == 1:
            a, b = sorted(rng.sample(range(1, 15), 2))
            p = f"List the integers from {a} to {b} inclusive, separated by commas, with nothing else."
            ans = ",".join(str(x) for x in range(a, b + 1))
        elif kind == 2:
            w = rng.choice(words)
            p = f"Write the word '{w}' backwards. Output only the reversed word."
            ans = w[::-1]
        elif kind == 3:
            ws = rng.sample(words, 3)
            p = (f"Sort these words alphabetically and output them separated by single spaces, "
                 f"nothing else: {', '.join(ws)}")
            ans = " ".join(sorted(ws))
        else:
            w = rng.choice(words)
            p = f"How many letters are in the word '{w}'? Answer with a number."
            ans = str(len(w))
        if p in SEEN:
            continue
        SEEN.add(p)
        i += 1
        out.append({"id": f"inst-{i:03d}", "skill": "instruction_following", "prompt": p,
                    "check": "numeric" if str(ans).isdigit() else "exact", "answer": ans})
    return out


def make_recall(n: int) -> list[dict]:
    facts = [
        ("What is the capital of France?", "paris"),
        ("What is the capital of Japan?", "tokyo"),
        ("What is the capital of Canada?", "ottawa"),
        ("What is the capital of Australia?", "canberra"),
        ("What is the capital of Brazil?", "brasilia"),
        ("What is the capital of Egypt?", "cairo"),
        ("What is the capital of India?", "new delhi"),
        ("What is the capital of Italy?", "rome"),
        ("What is the capital of Kenya?", "nairobi"),
        ("What is the capital of South Korea?", "seoul"),
        ("What is the chemical symbol for gold?", "au"),
        ("What is the chemical symbol for iron?", "fe"),
        ("What is the chemical symbol for sodium?", "na"),
        ("What is the chemical symbol for potassium?", "k"),
        ("What is the chemical symbol for lead?", "pb"),
        ("What is the chemical symbol for silver?", "ag"),
        ("What is the chemical symbol for tin?", "sn"),
        ("What is the chemical symbol for helium?", "he"),
        ("How many planets are in our solar system?", "8"),
        ("How many continents are there on Earth?", "7"),
        ("How many sides does a hexagon have?", "6"),
        ("How many degrees are in a right angle?", "90"),
        ("How many minutes are in three hours?", "180"),
        ("How many days are in a leap year?", "366"),
        ("How many legs does a spider have?", "8"),
        ("How many strings does a standard violin have?", "4"),
        ("Who wrote the play Romeo and Juliet?", "shakespeare"),
        ("Who wrote the novel Pride and Prejudice?", "jane austen"),
        ("Who painted the Mona Lisa?", "leonardo da vinci"),
        ("Who developed the theory of general relativity?", "einstein"),
        ("Which planet is known as the Red Planet?", "mars"),
        ("Which planet is closest to the Sun?", "mercury"),
        ("What is the largest planet in our solar system?", "jupiter"),
        ("What is the largest ocean on Earth?", "pacific"),
        ("What is the longest river in South America?", "amazon"),
        ("What is the tallest mountain on Earth above sea level?", "everest"),
        ("What gas do plants primarily absorb for photosynthesis?", "carbon dioxide"),
        ("What gas makes up most of Earth's atmosphere?", "nitrogen"),
        ("What is the chemical formula for water?", "h2o"),
        ("What is the chemical formula for table salt?", "nacl"),
        ("At what temperature does water boil at sea level, in Celsius?", "100"),
        ("At what temperature does water freeze at sea level, in Celsius?", "0"),
        ("What is the square root of 144?", "12"),
        ("What is 2 to the power of 10?", "1024"),
        ("How many bits are in a byte?", "8"),
        ("What does CPU stand for? Answer in full.", "central processing unit"),
        ("In which year did the Apollo 11 mission land humans on the Moon?", "1969"),
        ("In which year did World War II end?", "1945"),
        ("What is the currency of Japan?", "yen"),
        ("What is the currency of the United Kingdom?", "pound"),
    ]
    assert len(facts) >= n
    out = []
    for i, (q, a) in enumerate(facts[:n]):
        suffix = " Answer with a number." if a.isdigit() else " Answer with just the name or word."
        out.append({"id": f"recl-{i:03d}", "skill": "recall", "prompt": q + suffix,
                    "check": "numeric" if a.isdigit() else "exact", "answer": a})
    return out


def make_extraction(n: int) -> list[dict]:
    out = []
    i = 0
    while i < n:
        kind = i % 5
        name = rng.choice(NAMES)
        if kind == 0:  # invoice field
            inv, amt = rng.randint(1000, 9999), rng.randint(20, 900)
            text = (f"INVOICE #{inv} | Billed to: {name} Patel | Amount due: ${amt}.00 | "
                    f"Due date: 2026-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}")
            p = f"From this text, extract the amount due in dollars (number only): {text}"
            ans, chk = str(amt), "numeric"
        elif kind == 1:  # log line status
            code = rng.choice([200, 301, 404, 500, 503])
            text = (f"10.0.{rng.randint(0, 9)}.{rng.randint(1, 99)} - - "
                    f"\"GET /api/v1/runs HTTP/1.1\" {code} {rng.randint(100, 9999)}")
            p = f"From this log line, extract the HTTP status code (number only): {text}"
            ans, chk = str(code), "numeric"
        elif kind == 2:  # email sender
            user = name.lower()
            dom = rng.choice(["example.com", "mail.org", "corp.net"])
            text = (f"From: {user}@{dom}\nTo: team@corp.net\nSubject: weekly sync moved\n"
                    f"Body: moving the sync to Thursday.")
            p = f"From this email header, extract the sender's address only:\n{text}"
            ans, chk = f"{user}@{dom}", "exact"
        elif kind == 3:  # quantity in prose
            qty = rng.randint(3, 99)
            item = rng.choice(ITEMS)
            text = (f"The shipment arriving Tuesday contains {qty} {item}, "
                    f"two pallets, and one damaged crate that will be returned.")
            p = f"From this sentence, how many {item} are in the shipment? Number only: {text}"
            ans, chk = str(qty), "numeric"
        else:  # ID token
            token = f"RUN-{rng.randint(100, 999)}-{rng.choice('ABCDEF')}{rng.randint(10, 99)}"
            text = (f"Deployment failed. Ticket {token} was opened automatically; "
                    f"see the runbook before retrying.")
            p = f"Extract the ticket ID from this message, output it exactly and nothing else: {text}"
            ans, chk = token.lower(), "exact"  # harness lowercases both sides
        if p in SEEN:
            continue
        SEEN.add(p)
        i += 1
        out.append({"id": f"extr-{i:03d}", "skill": "extraction", "prompt": p,
                    "check": chk, "answer": ans})
    return out


def make_code(n: int) -> list[dict]:
    out = []
    i = 0
    while i < n:
        kind = i % 5
        if kind == 0:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            code = f"x = {a}\nfor i in range({b}):\n    x += i\nprint(x)"
            ans = str(a + sum(range(b)))
        elif kind == 1:
            lst = rng.sample(range(1, 30), 5)
            code = f"nums = {lst}\nprint(max(nums) - min(nums))"
            ans = str(max(lst) - min(lst))
        elif kind == 2:
            s = rng.choice(["platform", "curriculum", "gateway", "scheduler", "annealing"])
            k = rng.randint(2, 4)
            code = f"s = '{s}'\nprint(len(s[::{k}]))"
            ans = str(len(s[::k]))
        elif kind == 3:
            a = rng.randint(10, 60)
            b = rng.randint(2, 9)
            code = f"print({a} // {b} + {a} % {b})"
            ans = str(a // b + a % b)
        else:
            lst = sorted(rng.sample(range(1, 20), 4))
            code = f"xs = {lst}\nprint(sum(x for x in xs if x % 2 == 0))"
            ans = str(sum(x for x in lst if x % 2 == 0))
        p = f"What does this Python program print? Answer with the output only.\n```python\n{code}\n```"
        if p in SEEN:
            continue
        SEEN.add(p)
        i += 1
        out.append({"id": f"code-{i:03d}", "skill": "code", "prompt": p,
                    "check": "numeric", "answer": ans})
    return out


def main() -> None:
    items = (make_math(50) + make_reasoning(50) + make_instruction(50)
             + make_recall(50) + make_extraction(50) + make_code(50))
    assert len(items) == 300
    prompts = [it["prompt"] for it in items]
    assert len(set(prompts)) == 300, "duplicate prompts generated"
    for it in items:
        assert it["check"] in ("exact", "numeric") and it["answer"], it

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    per_skill = {}
    for it in items:
        per_skill[it["skill"]] = per_skill.get(it["skill"], 0) + 1
    print(f"wrote {len(items)} items to {OUT}")
    print(f"per skill: {per_skill}")
    print(f"sha256: {digest}")
    print("Record this sha256 in config/protocol.yaml. The set is now FROZEN.")


if __name__ == "__main__":
    main()
