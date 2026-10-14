# PhotonicsAI — Homework 1 (code for Tasks 5 and 6)

Erik Mirzoyan. Course: PhotonicsAI — AI-Driven Science and Autonomous Laboratories.

```
data/get_data.py              downloads SNLS80mV.mat (official site, GitHub mirror as fallback; MD5 checked)
task5_pinn_silverbox/         Task 5: physics-informed neural ODE for the Silverbox oscillator
task6_agentic_system/         Task 6: four-role LangGraph system that is given Task 5
report/                       LaTeX source and PDF of the solution
```

## Setup
Python 3.11–3.13. On Windows (PowerShell, or the PyCharm terminal):
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe data/get_data.py
```
On Linux/macOS: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python data/get_data.py`.
In PyCharm, open the folder and choose `.venv` as the project interpreter; the commands below then run as `python ...`.

The agent runs need a language model, either
* an API key: set `ANTHROPIC_API_KEY` (or another LangChain provider with `--model provider:model`), or
* no key: the Claude Code CLI logged in with a Claude subscription (`claude auth login`), and `--model claude-cli:claude-sonnet-5-5`
  (`task6_agentic_system/claude_cli.py`). All runs in the reports were made this way. If `claude` is not on PATH, set
  `CLAUDE_CLI` to its path.

The offline tests need no model and no key.

## Task 5
```
cd task5_pinn_silverbox
python sweep.py          # all architectures x training settings, physics ablation, references (~1 h on one CPU core)
python make_figures.py   # figures/, results/kept.json, ../report/task5_results.tex
```
One configuration: `python silverbox_pinn.py --json '{"name":"A2_h3","hidden":[64,64],"act":"tanh","rollout":256,"physics_weight":1.0}'`
(the PINN runs read the identified coefficients from `results/R_duffing.json`, produced by the `R_duffing` run).

Result: kept version A2_h5 (64x64 tanh, 4481 weights), free-run RMSE **0.422 mV** on the test record and 0.407 mV on the
training record (zero predictor: 60.49 mV). Every run is listed in `results/summary.md`.

## Task 6
```
cd task6_agentic_system
python tests/offline_smoke_test.py                          # wiring test with a scripted model (no key needed)
python run_task5.py --model claude-cli:claude-sonnet-5-5    # the four roles solve Task 5 (or --model anthropic:claude-sonnet-5-5 with a key)
python render_transcript.py runs/<timestamp> --tex ../report/task6_run.tex
```
Result of the reported run (`runs/20261014-001143/`): the script written and run by the executor printed a free-run RMSE
of **1.009 mV** on the test record and 1.446 mV on the training record.

## Report
```
cd report
pdflatex homework1.tex
pdflatex homework1.tex
```

