---
title: Monitoring data drift and LLM quality with Evidently
date: 2026-10-20
summary: A runnable Evidently 0.7 walkthrough covering tabular drift reports, LLM output checks with descriptors, CI pass/fail gates and scheduled runs
tags: MLOps, Evaluation, Python
---

A model that passed every offline test can still get worse in production without anything crashing. The inputs change, the relationship between inputs and outcomes changes, or an LLM starts giving longer, vaguer or off-brand answers after a prompt or model update. None of that shows up in uptime dashboards.

This post is the monitoring setup I use for both cases with [Evidently](https://github.com/evidentlyai/evidently), an open-source Python library for evaluating and monitoring ML and LLM systems. Every code block below is complete and I ran it end to end before publishing (versions are pinned in the install step).

## Why monitor at all

There are three different problems hiding under "the model got worse":

- **Data drift.** The distribution of inputs moves away from what the model was trained on. A new customer segment, a change in an upstream form, a sensor recalibration. The model may still be right, but you are now outside the data it learnt from.
- **Concept drift.** The relationship between inputs and the target changes. Same inputs, different correct answer. You usually only see this once ground-truth labels arrive, which can be days or weeks later.
- **LLM output quality.** For generative systems there is often no single correct label. What you can track is the behaviour of the outputs: length, tone, refusals, whether answers contain things they should or should not, and judgements from a second model.

Drift detection gives you an early warning before labels arrive. Text checks give you a cheap, repeatable read on LLM behaviour. Neither replaces measuring real quality once you have labels or human feedback, but both catch problems earlier.

## Install

Evidently's Python API changed substantially in 2025. Version 0.7 introduced a new core API (`Report`, `Dataset`, `DataDefinition`, presets in `evidently.presets`, descriptors for text), and the old API moved to `evidently.legacy`. Many tutorials online still use the old imports, so pin the version:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "evidently==0.7.23" "pandas<3" scikit-learn
```

I pin `pandas<3` because, in my testing, `evidently==0.7.23` failed on import with pandas 3.1 (`ImportError: cannot import name 'infer_dtype_from_object'`). It works with pandas 2.3. Check the release notes before you lift that pin.

The text descriptors for sentiment and word matching use NLTK data (`vader_lexicon` and `wordnet`). Evidently tries to download these on first use. If your CI runners have no outbound internet, download them once into the image:

```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('wordnet')"
```

## A tabular drift report

To keep this runnable without downloads I use the diabetes dataset bundled with scikit-learn. I split it into a reference batch (think "training data" or "last month") and a current batch ("today's traffic"), then deliberately shift two columns so there is something to find.

Save this as `monitor_drift.py`:

```python
import json
import sys

import numpy as np
from sklearn.datasets import load_diabetes

from evidently import Dataset, DataDefinition, Report
from evidently.metrics import DriftedColumnsCount, MissingValueCount, RowCount, ValueDrift
from evidently.presets import DataDriftPreset
from evidently.tests import eq, gte, lt

# 1. Build reference and current batches from a bundled sklearn dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)
df = X.assign(target=y)

rng = np.random.default_rng(42)
idx = rng.permutation(len(df))
reference_df = df.iloc[idx[:221]].reset_index(drop=True)
current_df = df.iloc[idx[221:]].reset_index(drop=True)

# Simulate a shift in two inputs so there is something to find
current_df["bmi"] = current_df["bmi"] + 0.03
current_df["s5"] = current_df["s5"] * 1.5

# 2. Tell Evidently what each column is
schema = DataDefinition(
    numerical_columns=[c for c in df.columns if c != "sex"],
    categorical_columns=["sex"],
)
reference = Dataset.from_pandas(reference_df, data_definition=schema)
current = Dataset.from_pandas(current_df, data_definition=schema)

# 3. Run the drift report (for humans)
report = Report([DataDriftPreset()])
snapshot = report.run(current_data=current, reference_data=reference)
snapshot.save_html("drift_report.html")
snapshot.save_json("drift_report.json")

# 4. Read the per-column results programmatically
result = json.loads(snapshot.json())
for metric in result["metrics"]:
    if metric["metric_name"].startswith("DriftedColumnsCount"):
        print("Drifted columns:", metric["value"])
    elif metric["metric_name"].startswith("ValueDrift"):
        cfg = metric["config"]
        print(f'{cfg["column"]:>8}  {cfg["method"]:<16} '
              f'score={metric["value"]:.4f}  threshold={cfg["threshold"]}')

# 5. Run the gate (for machines)
gate = Report(
    [
        DriftedColumnsCount(share_tests=[lt(0.3), lt(0.1, is_critical=False)]),
        ValueDrift(column="target"),
        MissingValueCount(column="bmi", tests=[eq(0)]),
        RowCount(tests=[gte(100)]),
    ],
    include_tests=True,
)
gate_result = json.loads(gate.run(current_data=current, reference_data=reference).json())

for t in gate_result["tests"]:
    print(f'[{t["status"]}] {t["name"]}')

failed = [t for t in gate_result["tests"] if t["status"] == "FAIL"]
sys.exit(1 if failed else 0)
```

A few things worth knowing about the API:

- `DataDefinition` is where you declare column types. Evidently can infer them, but being explicit avoids surprises. Here `sex` is stored as two scaled float values, so I declare it categorical.
- `report.run()` returns a **snapshot**. The report is the configuration; the snapshot is one computed result. You save, export or upload snapshots.
- `snapshot.json()` gives a JSON string with `metrics` and `tests` lists. `snapshot.dict()` returns the same structure as Python objects, but test statuses are enums there, so I parse the JSON when I want plain strings.

## Reading the drift results

Run it with `python monitor_drift.py`. The per-column part of the output looks like this:

```text
Drifted columns: {'count': 2.0, 'share': 0.18181818181818182}
     age  K-S p_value      score=0.6899  threshold=0.05
     bmi  K-S p_value      score=0.0001  threshold=0.05
      bp  K-S p_value      score=0.8403  threshold=0.05
      ...
      s5  K-S p_value      score=0.0014  threshold=0.05
  target  K-S p_value      score=0.2716  threshold=0.05
     sex  Z-test p_value   score=0.2153  threshold=0.05
```

How to read it:

- Evidently picks a statistical test per column based on type and sample size. With 221 rows it used the Kolmogorov-Smirnov test for numerical columns and a Z-test for the binary `sex` column. Both report a **p-value**, so a column counts as drifted when the score falls **below** the threshold of 0.05. `bmi` and `s5` are flagged, which is exactly the two columns I shifted.
- When I reran the same preset on a synthetic dataset with 5,000 rows, it switched to normalised Wasserstein distance for numerical columns and Jensen-Shannon distance for categorical ones, each with a threshold of 0.1. Those are **distances**, so drift means the score is **above** the threshold. Do not compare scores across methods without checking which kind you have.
- `DriftedColumnsCount` gives the count and share of drifted columns. The preset's default `drift_share` is 0.5: the whole dataset is flagged as drifted if half or more of the columns drift.
- The target column is included. In production you often will not have the target yet, but if you log predictions, put them through the same check. Prediction drift is a useful early proxy for concept drift.

The HTML file is the part I actually look at when something is flagged: it shows the reference and current distributions side by side for every column.

You can override the method and threshold per metric. For example, if your team is used to the population stability index:

```python
ValueDrift(column="bmi", method="psi", threshold=0.1)
```

## The gate: turning checks into pass/fail tests

The second report in the script is the one CI and schedulers care about. Each metric can carry test conditions from `evidently.tests` (`eq`, `not_eq`, `lt`, `lte`, `gt`, `gte`, `is_in`, `not_in`), and `include_tests=True` on the `Report` turns them on. Some metrics also add sensible default tests on their own: `ValueDrift(column="target")` gets an automatic "no drift" test.

The output for this run:

```text
[SUCCESS] Share of Drifted Columns: Less 0.300
[WARNING] Share of Drifted Columns: Less 0.100
[SUCCESS] Value Drift for column target
[SUCCESS] Column 'bmi' missing values: Equal 0.000
[SUCCESS] Row count in dataset: Greater or Equal 100.000
```

Two details that matter in practice:

- **Warnings versus failures.** `lt(0.1, is_critical=False)` produces `WARNING` instead of `FAIL`. I use that for a tighter "look at this" threshold alongside a looser "stop the pipeline" one. Here 18% of columns drifted, so the warning fired but the gate passed.
- **Why two reports.** If you put `DataDriftPreset()` in a report with `include_tests=True`, it adds its own default tests, including a per-column drift test. With version 0.7.23, passing `include_tests=False` to the preset did not switch those off when the report-level flag was on. Per-column tests are strict enough that one noisy column fails the run, so I keep the visual preset in one report and the explicit, tuned conditions in a separate gate report.

The script exits with code 1 when any test fails, which is all a CI system or scheduler needs.

## Evaluating LLM outputs with descriptors

For text, Evidently uses **descriptors**: functions that score each row and add the result as a new column. Some are deterministic (length, regex, word lists), some use small models (sentiment), and some call an LLM as a judge. You attach them when you build the `Dataset`, and you can attach tests to each one.

Here is a small set of prompt and response pairs from a hypothetical support assistant. Save it as `eval_llm.py`:

```python
import json
import sys

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import (
    DoesNotContain,
    IncludesWords,
    RegExp,
    Sentiment,
    TestSummary,
    TextLength,
)
from evidently.metrics import CategoryCount
from evidently.presets import TextEvals
from evidently.tests import eq, gte, lte

rows = [
    ("How do I reset my password?",
     "Go to Settings, then Security, and choose Reset password. You will get an email link."),
    ("What is your refund window?",
     "You can request a refund within 30 days of purchase from the Orders page."),
    ("Can you share another customer's address?",
     "I'm sorry, I can't share personal information about other customers."),
    ("Do you support SSO?",
     "Yes. We support SAML 2.0 single sign-on on the Business plan. "
     "Contact support@example.com to enable it."),
    ("Is the API rate limited?",
     "Yes, the API is rate limited. See the docs for current limits."),
]
df = pd.DataFrame(rows, columns=["question", "answer"])

eval_dataset = Dataset.from_pandas(
    df,
    data_definition=DataDefinition(text_columns=["question", "answer"]),
    descriptors=[
        TextLength("answer", alias="Length",
                   tests=[lte(300, alias="Under 300 chars")]),
        Sentiment("answer", alias="Sentiment",
                  tests=[gte(-0.25, alias="Not negative")]),
        IncludesWords("answer", words_list=["sorry", "apologize"], alias="Refusal"),
        RegExp("answer", reg_exp=r"[\w.+-]+@[\w-]+\.[\w.]+", alias="Has email"),
        DoesNotContain("answer", items=["as an AI language model"],
                       case_sensitive=False, alias="No boilerplate",
                       tests=[eq(True, alias="No AI boilerplate")]),
        TestSummary(success_all=True, alias="All tests passed"),
    ],
)

scored = eval_dataset.as_dataframe()
print(scored.drop(columns=["question", "answer"]).to_string())

# Report for humans: per-descriptor distributions plus row-level test results
report = Report([TextEvals()], include_tests=True)
snapshot = report.run(eval_dataset, None)
snapshot.save_html("llm_eval.html")

# Gate for machines: at least 80% of rows must pass every row-level check
gate = Report(
    [CategoryCount(column="All tests passed", category=True, share_tests=[gte(0.8)])],
    include_tests=True,
)
gate_result = json.loads(gate.run(eval_dataset, None).json())

for t in gate_result["tests"]:
    print(f'[{t["status"]}] {t["description"]}')

sys.exit(1 if any(t["status"] == "FAIL" for t in gate_result["tests"]) else 0)
```

The scored table it printed:

```text
   Length  Under 300 chars  Sentiment  Not negative  Refusal  Has email  No boilerplate  No AI boilerplate  All tests passed
0      85             True     0.3400          True    False      False            True               True              True
1      73             True     0.0000          True    False      False            True               True              True
2      68             True    -0.2933         False     True      False            True               True             False
3     103             True     0.6597          True    False       True            True               True              True
4      62             True     0.2023          True    False      False            True               True              True
[SUCCESS] Column 'All tests passed' categories 'True': Actual value 0.800 >= 0.800
```

Notes on what this shows:

- Each descriptor becomes a column, and each test becomes a `True`/`False` column next to it. `TestSummary` combines the tests defined **before** it in the list, so put it last.
- `RegExp` here flags answers that include an email address. In a real system that might be fine for a support address and a problem for anything else, which is why I track it as a signal rather than a test.
- Row 2 is a correct refusal, but VADER sentiment scores the apology as negative and it fails the "Not negative" test. This is the most common trap with sentiment checks: polite refusals look negative. I either exclude refusals from the sentiment test or use it only as a trend line, not a gate.
- Like the drift preset, `TextEvals` with `include_tests=True` generates its own tests that require every row to pass each descriptor test. That is useful in the HTML report but too strict as a gate on real traffic, so the gate report checks the **share** of passing rows instead. The threshold of 0.8 is illustrative; set yours from what you see on a baseline sample.

## Adding an LLM-as-judge descriptor

Deterministic checks cannot tell you whether an answer actually addresses the question. For that, Evidently has built-in judges (`DeclineLLMEval`, `CorrectnessLLMEval`, `FaithfulnessLLMEval`, `ToxicityLLMEval` and others) and a generic `LLMEval` with a prompt template. They need a provider API key; with the OpenAI provider that is the `OPENAI_API_KEY` environment variable. I did not run this block in my test environment because it calls a paid API, but the constructor signatures match version 0.7.23:

```python
from evidently.descriptors import DeclineLLMEval, LLMEval
from evidently.llm.templates import BinaryClassificationPromptTemplate

relevance = BinaryClassificationPromptTemplate(
    criteria="""An answer is RELEVANT if it directly addresses the user's question.
An answer is OFF_TOPIC if it ignores the question or talks about something else.
The user's question was: {question}""",
    target_category="OFF_TOPIC",
    non_target_category="RELEVANT",
    uncertainty="unknown",
    include_reasoning=True,
)

eval_dataset.add_descriptors(descriptors=[
    DeclineLLMEval("answer", alias="Declined"),
    LLMEval("answer", template=relevance, provider="openai", model="gpt-4o-mini",
            additional_columns={"question": "question"}, alias="Relevance"),
])
```

`additional_columns` maps a dataframe column to a placeholder in the criteria, so the judge sees the question as well as the answer. Before trusting a judge as a gate, label 30 to 50 rows by hand and check how often the judge agrees with you. If agreement is poor, fix the criteria wording before you change any thresholds.

## Saving, scheduling and the dashboard

The scripts already write HTML and JSON. To run the drift check daily, the simplest option is cron, replacing the synthetic data with a read from your inference logs:

```bash
# m h dom mon dow  command
0 6 * * * cd /opt/monitoring && .venv/bin/python monitor_drift.py >> logs/drift.log 2>&1
```

If you already run Airflow, wrap the script in a task so a failing gate shows up as a failed task with retries and alerting. This uses the Airflow 3 Task SDK imports:

```python
from datetime import datetime

from airflow.sdk import dag, task


@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False)
def drift_monitoring():
    @task
    def run_drift_check():
        import subprocess
        subprocess.run(
            ["/opt/monitoring/.venv/bin/python", "/opt/monitoring/monitor_drift.py"],
            check=True,
        )

    run_drift_check()


drift_monitoring()
```

HTML files are fine for one-off investigations, but you want trends over time. Evidently ships a self-hosted UI that reads snapshots from a workspace:

```python
from evidently.ui.workspace import Workspace

ws = Workspace.create("workspace")
project = ws.create_project("Diabetes model monitoring")
project.description = "Daily drift checks"
project.save()

ws.add_run(project.id, snapshot, include_data=False)
```

Then start the UI with `evidently ui --workspace ./workspace` and open `http://127.0.0.1:8000`. In a scheduled job you would create the project once, store its ID, and call `add_run` after each report. Evidently Cloud offers the same flow through `CloudWorkspace` if you prefer not to host it. `include_data=True` uploads the row-level data too, which is handy for LLM evals but worth thinking about if the text contains personal data.

## Limitations and practical thresholds

- **Drift is not the same as degradation.** A feature can drift without hurting accuracy, and accuracy can fall with no visible drift (concept drift). Treat drift as a reason to look, and confirm with performance metrics once labels arrive, for example with the classification or regression presets.
- **Sample size changes everything.** With large batches, p-value tests flag tiny, harmless differences. Evidently already switches to distance metrics above roughly 1,000 rows for this reason. Keep your current window big enough to be stable but small enough to react, for example one day of traffic.
- **Start loose, then tighten.** I start with the defaults (0.05 for p-values, 0.1 for distances), a dataset-level gate such as a drifted share under 0.3, and a non-critical warning at a lower level. Then I adjust per column after a few weeks of baseline runs.
- **Text descriptors are proxies.** Length, sentiment and regex checks are cheap and deterministic, which makes them good gates. They do not measure correctness. LLM judges get closer but add cost, latency and their own errors, so calibrate them against human labels and re-check after any judge model change.
- **Pin versions.** The 0.7 API is very different from 0.4 and earlier, and behaviour such as default tests can change between minor releases. Pin Evidently in your monitoring image and re-run your gate on a known batch when you upgrade.

## Sources

- [Evidently documentation](https://docs.evidentlyai.com/)
- [Evidently LLM quickstart](https://docs.evidentlyai.com/quickstart_llm)
- [Evidently descriptors](https://docs.evidentlyai.com/docs/library/descriptors)
- [Evidently data definition](https://docs.evidentlyai.com/docs/library/data_definition)
- [Evidently migration guide (0.6 to 0.7)](https://docs.evidentlyai.com/faq/migration)
- [Evidently on GitHub](https://github.com/evidentlyai/evidently)
- [Evidently on PyPI](https://pypi.org/project/evidently/)
- [scikit-learn diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
