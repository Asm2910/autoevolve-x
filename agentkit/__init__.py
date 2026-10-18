"""The agent platform: the shared machinery every agent runs on.

  budget       hard limits on LLM calls, tokens, dollars, time and experiments, plus a kill switch
  guardrails   untrusted-text handling and prompt-injection defence
  data_access  the agents' view of the data; the sealed test split is not reachable
  tracing      structured trace log and Prometheus metrics for every agent step
  harness      the base Agent: typed contract, timeout, retries, tool allow-list
  context      what goes into a prompt: token budget, history compaction, versioned prompts
  llm          token accounting and record/replay doubles so missions run in CI without an API key

None of this makes a language-model system safe in an absolute sense. It limits what a mistake
or a hostile input can do, and the tests in tests/agent_evals check those limits.
"""
