// Free-model aggregator: one server, two client wires, ten free providers.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts`
// with Node 22's standard library alone. It reads the brief, probes each provider's
// catalog route in the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (OpenAI shape for an Authorization
//      header, Anthropic shape for x-api-key), POST /v1/chat/completions and POST
//      /v1/messages, each in JSON or server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says done.
//      A process still running when the wall ends scores zero, so it never waits for
//      anything else.
//
// Routing: for each request, the router ranks every free model that meets it by the quality
// table's rating, best first. The server tries them in that order — one provider call at a
// time, never two at once — and answers with the first that answers: a refusal (402, 429,
// 5xx, or a call that never comes back) is counted and the next candidate is tried, so a
// provider whose budget is spent takes no request down with it. Exactly one successful
// provider call answers a request; a second success is never attempted.
//
// `node main.ts --list` prints the same catalog JSON the register body carries and exits 0
// before any server starts. Run outside a trial, with no brief to read, it prints
// {"providers":[]} and exits 0. The catalog is built once, as one object, and used for both.
//
// Every choice is a function of the brief, the requests and the providers' answers: no wall
// clock and no unseeded randomness, so a replay sees the same answers and makes the same
// choices, and the state export matches byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type Probe,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamEvents,
  anthropicStreamParts,
  anthropicRequestToGemini,
  anthropicAnswerToOpenAI,
  anthropicRequestToOpenAI,
  collectSse,
  geminiStreamParts,
  geminiAnswerToAnthropic,
  geminiAnswerToOpenAI,
  openAIError,
  openAIModelList,
  openAIStreamEvents,
  openAIStreamParts,
  openAIAnswerToAnthropic,
  openAIRequestToAnthropic,
  openAIRequestToGemini,
  type AnswerParts,
  type ServedModel,
} from './lib/wire.ts';
import { Router, type Candidate } from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let router: Router | null = null;
/** Client API requests arrived so far: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
  router = new Router(probes, brief.ratings);
}

/** The free models across the providers that answered, each id once, in brief order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      // The catalog's free flag is the truth here; the router holds the same free set.
      const entry = catalog.providers.find((p) => p.id === probe.provider.id);
      const listed = entry?.models.find((m) => m.id === model.id);
      if (listed === undefined || !listed.free) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.workload[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The request one candidate wants, in its own wire. */
function upstreamFor(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const base = joinBase(ENV_URL, candidate.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.provider.auth, candidate.provider.key),
  };
  const wire = candidate.provider.wire;
  if (wire === 'openai') {
    const upstreamBody =
      clientApi === 'openai'
        ? {
            ...body,
            model: candidate.model.id,
            ...(stream
              ? { stream_options: { ...(isRecord(body.stream_options) ? body.stream_options : {}), include_usage: true } }
              : {}),
          }
        : anthropicRequestToOpenAI(body, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    const upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIRequestToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  // Gemini: the model rides in the path, and the stream asks for :streamGenerateContent.
  const path = stream ? ':streamGenerateContent' : ':generateContent';
  const upstreamBody = clientApi === 'openai' ? openAIRequestToGemini(body, candidate.model.id) : anthropicRequestToGemini(body, candidate.model.id);
  if (stream) headers.accept = 'text/event-stream';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${path}`, headers, body: upstreamBody };
}

/** Copy a provider's event stream to the client as it arrives (same wire both ends). */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** One provider's JSON answer, translated into the client's wire. */
function translateJson(clientApi: 'openai' | 'anthropic', wire: string, answer: Json, model: string): Json {
  if (clientApi === 'openai') {
    if (wire === 'openai') return answer;
    if (wire === 'anthropic') return anthropicAnswerToOpenAI(answer, model);
    return geminiAnswerToOpenAI(answer, model);
  }
  if (wire === 'anthropic') return answer;
  if (wire === 'openai') return openAIAnswerToAnthropic(answer, model);
  return geminiAnswerToAnthropic(answer, model);
}

/** One provider's SSE stream, collected and re-emitted in the client's wire. */
async function translateStream(
  clientApi: 'openai' | 'anthropic',
  wire: string,
  stream: ReadableStream<Uint8Array>,
  model: string,
): Promise<string> {
  const frames = await collectSse(stream);
  let parts: AnswerParts;
  if (wire === 'openai') parts = openAIStreamParts(frames);
  else if (wire === 'anthropic') parts = anthropicStreamParts(frames);
  else parts = geminiStreamParts(frames);
  return clientApi === 'openai' ? openAIStreamEvents(parts, model) : anthropicStreamEvents(parts, model);
}

/** Serve one client request: try the ranked candidates in order, answer with the first that
 * answers, and count every refusal on the way. Exactly one successful provider call answers
 * a request; a provider already tried for this request is tried again only after every
 * other candidate has refused, on another of its models.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonMode =
    clientApi === 'openai' &&
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  if (router === null) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no providers were probed');
  }
  const ordered = router.orderedFor(requires, jsonMode);
  if (ordered.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model meets this request');
  }
  const tried = new Set<string>();
  // First pass: one model per provider, best first. Second pass: the other models of the
  // providers already tried — a refusal may have been one scheduled call, not a spent budget.
  const passes: Candidate[][] = [[], []];
  for (const candidate of ordered) {
    if (tried.has(candidate.provider.id)) passes[1].push(candidate);
    else {
      tried.add(candidate.provider.id);
      passes[0].push(candidate);
    }
  }
  for (const pass of passes) {
    for (const candidate of pass) {
      const call = upstreamFor(candidate, clientApi, body, stream);
      let upstream: Response;
      try {
        upstream = await fetch(call.url, {
          method: 'POST',
          headers: call.headers,
          body: JSON.stringify(call.body),
          signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
        });
      } catch (err) {
        router.noteFailure(candidate.provider.id);
        log(`${candidate.provider.id}/${candidate.model.id} did not answer: ${(err as Error).message}`);
        continue;
      }
      if (!upstream.ok) {
        router.noteFailure(candidate.provider.id);
        log(`${candidate.provider.id}/${candidate.model.id} refused HTTP ${upstream.status}`);
        try {
          await upstream.text();
        } catch {
          // The body is spent either way.
        }
        continue;
      }
      // One success answers the request; nothing else is called for it.
      router.noteSuccess(candidate.provider.id);
      log(`${candidate.provider.id}/${candidate.model.id} answered`);
      const wire = candidate.provider.wire;
      if (stream && upstream.body !== null && clientApi === 'openai' && wire === 'openai') {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        await pipeStream(upstream.body, res);
        return;
      }
      if (stream && upstream.body !== null && clientApi === 'anthropic' && wire === 'anthropic') {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        await pipeStream(upstream.body, res);
        return;
      }
      if (stream && upstream.body !== null) {
        const text = await translateStream(clientApi, wire, upstream.body, candidate.model.id);
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(text);
        return;
      }
      const answer: unknown = await upstream.json();
      sendJson(res, 200, translateJson(clientApi, wire, isRecord(answer) ? answer : {}, candidate.model.id));
      return;
    }
  }
  log('every candidate refused; the request goes unanswered');
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model refused this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object
    // and stop before any server starts. A process whose stdout is not flushed loses the tail.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} probed providers`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
