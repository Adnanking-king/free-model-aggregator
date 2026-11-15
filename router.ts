// Routing: which free model answers a request, and what to try when a provider refuses.
//
// Every candidate is a free model that meets the request's requirements, ranked by the
// quality table's rating, best first, with provider and model id breaking ties so the order
// is the same on a replay. A request that needs JSON mode on the OpenAI wire can only go to
// a wire that can carry it (openai or gemini): the Anthropic wire has no JSON mode, so a
// provider there would echo it absent. (An Anthropic client request never needs a wire to
// carry JSON mode: its own wire has none, so any wire serves it as long as the model can.)
//
// Providers are budgeted per trial and refuse on a schedule the brief does not publish, so
// the router learns from what it sees: a provider that refuses twice running, with no
// success between, is demoted behind every provider that has not, until it answers again.
// One refusal is not a verdict — a provider refuses single calls on its own schedule and
// answers the next — but two running mean its budget is likely spent or it is out, and the
// calls after that are better spent on the next-best model. The demotion is a function of
// the provider answers seen so far, so a replay, which sees the same answers, demotes the
// same providers in the same order.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type ListedModel,
  type Probe,
  type ProviderEntry,
  type RatingsTable,
  type Requirement,
} from './brief.ts';

export interface Candidate {
  provider: ProviderEntry;
  model: ListedModel;
  rating: number;
}

/** Two refusals running, with no success between, demote a provider. */
const SUSPECT_AFTER_FAILURES = 2;

export class Router {
  private readonly ratings: RatingsTable | null;
  private readonly candidates: Candidate[];
  /** Refusals since each provider last answered, by provider id. */
  private readonly failuresSinceSuccess = new Map<string, number>();

  constructor(probes: readonly Probe[], ratings: RatingsTable | null) {
    this.ratings = ratings;
    const all: Candidate[] = [];
    for (const probe of probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingOfModel(ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        all.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
      }
    }
    // Best rating first; provider then model id break ties, deterministically.
    all.sort(
      (a, b) =>
        b.rating - a.rating ||
        (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
        (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
    );
    this.candidates = all;
  }

  /** Whether a provider is demoted: it refused twice running with no success between. */
  private isSuspect(providerId: string): boolean {
    return (this.failuresSinceSuccess.get(providerId) ?? 0) >= SUSPECT_AFTER_FAILURES;
  }

  /** The candidates that can serve a request, best first, demoted providers last. */
  orderedFor(requires: Requirement, jsonMode: boolean): Candidate[] {
    const fitting = this.candidates.filter((c) => {
      if (jsonMode && c.provider.wire === 'anthropic') return false;
      const row = ratingOfModel(this.ratings, c.provider.id, c.model.id);
      return meetsRequirements(requires, capabilitiesOf(c.model, row));
    });
    const fresh = fitting.filter((c) => !this.isSuspect(c.provider.id));
    const suspect = fitting.filter((c) => this.isSuspect(c.provider.id));
    return [...fresh, ...suspect];
  }

  /** How many times a provider has refused running, with no success between. */
  private failuresSince(providerId: string): number {
    return this.failuresSinceSuccess.get(providerId) ?? 0;
  }

  /** Record that a provider answered: it is trusted again. */
  noteSuccess(providerId: string): void {
    this.failuresSinceSuccess.set(providerId, 0);
  }

  /** Record that a provider refused (or could not be reached). */
  noteFailure(providerId: string): void {
    this.failuresSinceSuccess.set(providerId, (this.failuresSinceSuccess.get(providerId) ?? 0) + 1);
  }
}
