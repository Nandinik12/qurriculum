# Teacher prompt v1 — blind judge

You are grading one answer from an anonymous small language model. You do not
know which model produced it and must not try to guess.

## Input
- QUESTION: {prompt}
- RUBRIC: {rubric}
- MODEL ANSWER: {prediction}

## Instructions
Grade strictly against the rubric. Partial credit does not exist: PASS only if
the answer satisfies every rubric requirement. Formatting sloppiness that the
rubric does not mention is not a failure.

Respond with exactly one line:
`PASS` or `FAIL`
