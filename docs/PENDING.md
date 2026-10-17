# What is still pending

Last updated 2026-10-17.

## Quick actions for you

- [ ] **Grafana screenshot.** Start the stack, send traffic, open http://localhost:3000, and save a screenshot as `docs/images/grafana-dashboard.png`. Then uncomment the Grafana image line in `README.md`.
- [ ] **Your own dashboard screenshot.** `docs/images/dashboard-example.png` comes from a different run, so its numbers differ slightly from the 0.272 in the documents. Replace it with a screenshot of your own run of `python -m streamlit run dashboard/app.py`.
- [ ] **Check `.gitignore`.** Run `git check-ignore -v .env`. It must print a rule. See `SECURITY.md`.
- [ ] **Copyright holder.** `LICENSE` names "Abinesh Siva", taken from your email address. Edit it if you want a different name, or choose another licence.

## Verification that has not been done

- [ ] **GitHub Actions.** Confirm both jobs (lint and test, Docker build) are green on `main`. The Docker job could not be tested in the sandbox.
- [ ] **Grafana.** Confirm all nine panels render, especially the score heatmap.
- [ ] **Docker image tags.** `prom/prometheus:v3.5.0` and `grafana/grafana:12.1.0` were pinned from memory. Change the version if Docker cannot pull them.
- [ ] **Three inferred details** in the technical document: the meaning of `retrain_share` (0.05) in `monitoring/drift.py`, the `autoevolve-x-ablation` experiment name, and the exact wording of the data dictionary's column meanings, which were written from the code and column names.

## Engineering (optional, in priority order)

- [ ] Run the pipeline on a **real public fraud dataset**, to test the synthetic-data conclusions.
- [ ] Add **probability calibration**, so scores can be read as rates.
- [ ] Correct for **multiple comparisons** when many agent experiments share one validation set.
- [ ] Add **Alertmanager**, so alerts reach a person.
- [ ] Add a **data-ingestion step**, so a drift alert can trigger a retrain.
- [ ] Add **authentication** to the API.
- [ ] Decide whether the agent may propose **new features**, not only model settings.
