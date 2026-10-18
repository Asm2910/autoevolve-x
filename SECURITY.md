# Security notes

## The API key

The only secret in this project is the Anthropic API key, and it is only needed for the language-model proposer (`python -m scripts.run_agent --proposer llm`). Everything else, including the rule-based agent, the API, the dashboard and the tests, runs without it.

- The key is read from the environment variable `ANTHROPIC_API_KEY`, or from a local `.env` file that `scripts/run_agent.py` loads at start-up. A real environment variable always wins over `.env`.
- `.env` is never committed. `.env.example` shows the format and holds no value.
- Never paste the key into a chat, an issue or a commit message.

### Check your repository once

Run these from the repository root.

```bash
# 1. .env must be ignored: this should print a rule that mentions .env
git check-ignore -v .env

# 2. If it printed nothing, add the rule and commit it
echo ".env" >> .gitignore

# 3. Confirm no key was ever committed: this should print nothing
git log --all -p -S "sk-ant" --oneline | head
```

If step 3 prints anything, treat that key as leaked: revoke it in the Anthropic Console, create a new one, and remove the commit from history before pushing.

## Agent safety (guardrails)

The language model proposes; ordinary code validates, runs and decides. On top of that structure there are explicit limits, all in `agentkit/` and tested in `tests/agent_evals/`:

- **Budgets.** Each run has hard limits on language-model calls, tokens, estimated cost, wall-clock time and experiments (`agent_platform` in `configs/models.yaml`). The loop stops with a stated reason, and the fallback proposer does not hide a limit by switching to rules. The cost figure uses assumed prices, so set them to your plan's rates.
- **Kill switch.** `touch memory/STOP` stops a running agent before its next step. Delete the file to run again.
- **Sealed test data.** Agents read data only through `agentkit/data_access.py`, which refuses the test split. A test also scans the agent code to confirm nothing reads the test file directly.
- **Untrusted text.** Lessons, history and refusal messages are sanitised (control characters removed, length capped, angle brackets escaped), wrapped in `<untrusted_data>` tags, and anything that reads like an instruction to the model is replaced by a placeholder. This reduces risk but is not a guarantee. The real protection is that the model can only return an experiment spec, which is checked against allow-lists and bounds, and adoption is decided by a fixed statistical rule.
- **Traces.** Every agent step is written to `memory/traces/*.jsonl` with its duration, outcome and prompt version, so a run can be audited.
- **Not covered.** These controls do not protect the API key if you paste it somewhere, do not authenticate the prediction API, and cannot stop a human from running code that reads the test file.

## Other things worth knowing

- **Grafana.** The compose file allows anonymous viewing and uses `admin` for editing, with the password `admin` unless `GRAFANA_ADMIN_PASSWORD` is set. This is fine on a laptop. Set a real password before exposing port 3000 to any network.
- **The API has no authentication.** `/predict` accepts any request on port 8000. Put it behind an authenticating gateway before real use.
- **Container user.** The Docker image runs as a non-root user.
- **Data.** All data is synthetic. No personal data is stored anywhere in the repository.
- **Dependencies.** Versions are pinned in `requirements.txt` and `requirements-dev.txt`. Review and update them periodically.

## Reporting a problem

Open an issue on `Asm2910/autoevolve-x`, or email the maintainer at the address on the GitHub profile. Do not include secrets in the report.
