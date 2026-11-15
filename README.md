# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Adnanking-king/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Adnanking-king/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A quality-ordered aggregator for free AI models: one server that speaks the OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in server-sent events, over ten free-tier providers (OpenAI, Anthropic and Gemini wires). It ranks every free model by the quality table, serves each request with the best one that meets it, fails over when a provider's budget is spent or it refuses a call, and demotes a provider that refuses twice running until it answers again. `node main.ts --list` prints the catalog it found.

## Run it

```sh
node main.ts
```

The trial sets `MAKEFUN_PROMPT_FILE` (the brief: providers, quality table, workload) and `MAKEFUN_ENV_URL` (the env that serves the providers and the workload). The server probes each provider's catalog, registers its port and catalog with the env, serves the twenty requests, and exits 0 when the env says the workload is done. No provider key or account of yours is needed: the providers are the trial's simulated free tiers, keyed from the brief.

## Layout

- `main.ts` — the server: brief, probe, catalog, register, routes, failover loop, poll-and-exit.
- `lib/brief.ts` — the brief's three documents, the catalog probe, the catalog builder.
- `lib/router.ts` — rating-ordered candidates, with demotion after two refusals running.
- `lib/wire.ts` — the translations: OpenAI, Anthropic and Gemini requests and answers, JSON and SSE, in both directions.

---

<sub>Entered on [make.fun](https://make.fun/?ref=v9pvyfm7&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
