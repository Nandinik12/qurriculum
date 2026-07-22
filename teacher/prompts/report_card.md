# Teacher prompt v1 — weekly report card

Write the week-{week} report card for the qurriculum project as a draft blog
post. Audience: technical readers following a public experiment; tone: direct,
data-first, no hype.

## Input
- CURVES: {curves_jsonl_tail}          (all weeks so far, all three students)
- THIS WEEK'S SELECTION LOG: {selection_json}   (solutions, objectives, wall times, diffs)
- WEAKNESS VECTORS: {weakness_vectors}

## Structure
1. **Scoreboard** — three students' static-eval scores this week vs last, per skill.
2. **What the solvers disagreed about** — from the selection diffs: how many
   examples differed, what kinds (skills/difficulty), Jaccard overlap. This is
   the most interesting section; be concrete about *which* examples quantum
   picked that classical didn't.
3. **Solver stats** — objective values and wall times, including whether the
   quantum run used direct QPU or hybrid fallback.
4. **Honest read** — is any curve separation beyond noise yet? Never overclaim.
   "No separation yet" is a fine and publishable sentence.
5. **Next week** — which skills the weakness vectors will upweight.

Keep it under 800 words. Include the data table; the human adds the chart.
