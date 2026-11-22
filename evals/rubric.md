# LLM-as-judge rubric — Plate Coach

Used by `evals/compare_models.py`. The judge is **one fixed model for every comparison**
(`JUDGE_MODEL`, default `anthropic/claude-sonnet-5.5`), and that model is never among
the models under test, so no model grades its own answers. The judge receives the
task, the agent's output, and the **reference data** (the nutrition-database records
and the Healthy Score / projected scores the agent's own code computed), and returns
one integer score of 1–5 per criterion, as JSON.

## Criteria

| Criterion | Key | What the judge checks |
|---|---|---|
| Role relevance | `relevance` | Stays in the Plate Coach role: answers the meal-nutrition task asked, no off-topic content, no medical diagnosis or prescriptions. |
| Factual accuracy | `factual_accuracy` | Every nutrition statement agrees with the reference records (per 100 g values, which food is higher/lower). Qualitative claims (e.g. "carbs: high") fit the numbers. |
| Completeness | `completeness` | Covers everything the task asks: for a meal analysis, a qualitative profile and 2–4 concrete suggestions; for a question, a direct answer to every part of it. |
| Format adherence | `format` | Follows the required format: suggestions are short, soft ("you could…") and number-free (numbers come from code); answers are 2–5 sentences; structured outputs parsed correctly (no missing suggestions). |
| Data reliability | `data_reliability` | No invented numbers: any figure stated appears in, or follows directly from, the reference data; foods missing from the database are said to be missing rather than guessed. |

`overall` is the mean of the five criteria (computed in code, not by the judge).

## Scale (same for every criterion)

| Score | Meaning |
|---|---|
| 5 | Fully meets the criterion; nothing to correct. |
| 4 | Meets it, with one minor slip (a slightly vague phrase, one small omission). |
| 3 | Partly meets it: a noticeable gap or one clear error. |
| 2 | Mostly fails it: several errors or a major omission. |
| 1 | Fails it entirely (off-topic, wrong data, empty or unparseable output). |

## Judge output format

```json
{"relevance": 1-5, "factual_accuracy": 1-5, "completeness": 1-5,
 "format": 1-5, "data_reliability": 1-5, "comment": "<one sentence>"}
```

A run that fails (API error, timeout) is recorded with `status=error` and no scores;
it is counted against the model's reliability (success rate), not averaged as a 1.
