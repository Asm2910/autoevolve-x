# What is still pending

Last updated 2026-10-18.

## Quick actions for you

- [ ] **Grafana screenshots.** With the stack up and a mission run, save the agents dashboard as `docs/images/grafana-agents.png` and the API dashboard (after `scripts.generate_traffic`) as `docs/images/grafana-dashboard.png`. Then uncomment the two image lines in `README.md`.
- [ ] **Your own review-dashboard screenshot.** `docs/images/dashboard-example.png` is an older example. Replace it with a screenshot of your own run of `python -m streamlit run dashboard/app.py`.
- [ ] **Sealed-test confirmation for v2.** Run `python -m scripts.confirm_champion` once for telecom and once with `AUTOEVOLVE_DATASET=ieee`. Each refuses a second run. Then add the paired difference against logistic regression to the README results.
- [ ] **Check `.gitignore`.** `git check-ignore -v .env` must print a rule. See `SECURITY.md`.
- [ ] **Copyright holder.** `LICENSE` names "Abinesh Siva", taken from your email address. Edit it if you want a different name.

## Verification that has not been done

- [ ] **GitHub Actions.** Confirm both jobs are green on `main`, including the new mission and research smoke test and the Ray tests (the Ray tests were only run locally).
- [ ] **Docker.** The API image builds in CI. The full compose stack was checked by hand on one machine only, and the case where CatBoost is the served model was never run.
- [ ] **Language-model paths with a live model.** The proposer, the research answer (`scripts.ask --llm`) and the mission were exercised with scripted test doubles. Run each once with a real key and read the output.
- [ ] **Image tags.** `prom/prometheus:v3.5.0`, `grafana/grafana:12.1.0`, `prom/pushgateway:v1.11.1` and `ghcr.io/mlflow/mlflow:v3.17.1` are pinned versions. Change one if Docker cannot pull it.
- [ ] **Three inferred details** in the technical document: the meaning of `retrain_share` (0.05) in `monitoring/drift.py`, the `autoevolve-x-ablation` experiment name, and the wording of the data dictionary's column meanings.

## Known gaps, by design or by time

- The mission planner picks steps from the depth and does not interpret the goal text.
- The prompt-injection filter is pattern-based.
- Ray gave no speed-up on a 10-core machine for 7 small models. A real test of Ray's value needs larger models or more than one machine.
- Probability calibration was skipped.
- Alertmanager was skipped, so alerts do not reach a person.
- The API has no authentication.
- No data-ingestion step, so a drift alert cannot trigger a retrain.
- The agents dashboard keeps only the last run of each script.
- IEEE-CIS is one real dataset, sampled to 300,000 rows. Telecom data is synthetic.

## Left out on purpose

Kubernetes, Kafka, Spark, Airflow, GPUs, language-model fine-tuning, more than a dozen agents and production authentication are out of scope for this project.

## Possible next steps

- Let the agent propose new features, not only model settings.
- Correct for multiple comparisons across experiments that share one validation set.
- Add calibration, Alertmanager and ingestion.
- Run the arena on a larger dataset to see whether Ray starts to help.
