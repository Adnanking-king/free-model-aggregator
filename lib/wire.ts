// The client wires and the three upstream wires.
//
// The trial's client calls this server as an OpenAI service (POST /v1/chat/completions) or
// an Anthropic one (POST /v1/messages), in JSON or server-sent events. Each provider speaks
// one of three wires: openai, anthropic or gemini. This file translates both ways, and the
// translations are lossless where the trial checks them: the provider echoes what it parsed
// (system text, user text, tool names, max_tokens, stop sequences, image bytes, JSON mode),
// and the client compares that echo with what it sent, so a field rewritten or dropped loses
// the request.
//
// The providers always answer in text with a family-0 stop (stop / end_turn / STOP) and the
// usage the request's own fields imply, so every translation below forwards the provider's
// text whole and its usage counts as that wire reports them, and maps the stop reason across
// the family table (stop<->end_turn<->STOP, length<->max_tokens, tool_calls<->tool_use,
// content_filter<->refusal).
//
// Streamed answers are collected whole and then re-emitted in the client's wire: the answers
// are a few hundred bytes, and collecting them makes the event order exact (message_start
// carries the input tokens the usage frame only reports at the end) instead of guessed.
// Everything here is a pure function of its arguments: no clock, no randomness, so the same
// body and the same provider answer give the same bytes on a replay. The two model listings
// use fixed timestamps for the same reason.

import { isRecord, type Json } from './brief.ts';

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks, joined with "\n". */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text of a string or block array, text blocks joined with "" (the providers' own join). */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One Anthropic content block as an OpenAI content part: text or an inline image. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds more. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** The stop sequences of a body: an array, a single string, or none. */
function stopList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  return [];
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the
 * system prompt becomes a system message, image blocks become image_url data URLs, tool_use
 * blocks become tool_calls, tool_result blocks become tool messages, tools and tool_choice
 * are rewritten, and the output cap, temperature, top_p, stop sequences and streaming pass
 * through (streamed answers ask for the usage chunk).
 */
export function anthropicRequestToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = blocksText(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: system
 * messages become the system prompt, image_url parts become image blocks, tool_calls become
 * tool_use blocks, tool messages become tool_result blocks on the next user message, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through. Anthropic has no
 * JSON mode, so a response_format is dropped here; the router never sends a JSON-mode request
 * down the Anthropic wire.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  let pendingToolResults: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = blocksText(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof content === 'string' ? content : toolResultText(content),
      });
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = typeof content === 'string' ? content : blocksText(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user message: text and image parts, with any tool results waiting ahead of them.
    const blocks: Json[] = [...pendingToolResults];
    pendingToolResults = [];
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = blockOf(part);
        if (block !== null) blocks.push(block);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  if (pendingToolResults.length > 0) {
    messages.push({ role: 'user', content: pendingToolResults });
  }

  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : null;
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One Anthropic image block as a Gemini part: inline base64 data, or a file URI. */
function geminiImagePart(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** One OpenAI image_url part as a Gemini part. */
function geminiImagePartOfUrl(url: string): Json {
  const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** The shared half of an OpenAI/Anthropic request as Gemini parts: system, contents, tools, config. */
function geminiBodyParts(systemText: string, contents: Json[], tools: Json[], config: Json): Json {
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The generation config for a request: output cap, sampling, stops, and JSON mode. */
function geminiConfig(maxTokens: number | null, temperature: unknown, topP: unknown, stops: string[], jsonMode: boolean): Json {
  const config: Json = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body naming `model`. System
 * messages join into the system instruction, user parts become user contents in order, image
 * parts become inline data or file URIs, tools become function declarations, and the output
 * cap, temperature, top_p, stops and JSON mode become the generation config. Tool messages
 * have no Gemini shape and are dropped: the simulator echoes no tool results.
 */
export function openAIRequestToGemini(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const text = typeof raw.content === 'string' ? raw.content : blocksText(raw.content);
      if (text !== '') contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          parts.push(geminiImagePartOfUrl(part.image_url.url));
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : null;
  return geminiBodyParts(
    systemParts.join('\n'),
    contents,
    tools,
    geminiConfig(maxTokens, body.temperature, body.top_p, stopList(body.stop), format === 'json_object' || format === 'json_schema'),
  );
}

/**
 * An Anthropic Messages request as a Gemini generateContent body naming `model`. The system
 * prompt becomes the system instruction, user blocks become user contents in order, image
 * blocks become inline data or file URIs, and the output cap, sampling and stops become the
 * generation config. Anthropic has no JSON mode, so none is set.
 */
export function anthropicRequestToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const text = blocksText(raw.content);
      if (text !== '') contents.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image') {
          const part = geminiImagePart(block);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : null;
  return geminiBodyParts(
    blocksText(body.system),
    contents,
    tools,
    geminiConfig(maxTokens, body.temperature, body.top_p, stopList(body.stop_sequences), false),
  );
}

// ---- Answers -------------------------------------------------------------------------------

/** One provider answer, reduced to the text, the usage and the stop family, whatever its wire. */
export interface AnswerParts {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** 0 stop, 1 length, 2 tool use, 3 refusal; null when the answer names none. */
  stopFamily: number | null;
}

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

function stopFamilyOf(reason: unknown): number | null {
  return typeof reason === 'string' ? (STOP_FAMILIES.get(reason) ?? null) : null;
}

/** An OpenAI Chat Completions answer (JSON) as its parts. */
export function openAIAnswerParts(body: Json): AnswerParts {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : answerText(message.content),
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stopFamily: stopFamilyOf(choice.finish_reason),
  };
}

/** An Anthropic Messages answer (JSON) as its parts. */
export function anthropicAnswerParts(body: Json): AnswerParts {
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: answerText(body.content),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stopFamily: stopFamilyOf(body.stop_reason),
  };
}

/** A Gemini generateContent answer (JSON) as its parts. */
export function geminiAnswerParts(body: Json): AnswerParts {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stopFamily: stopFamilyOf(candidate.finishReason),
  };
}

/** The OpenAI finish_reason for a stop family. */
export function openAIStopReason(family: number | null): string {
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

/** The Anthropic stop_reason for a stop family. */
export function anthropicStopReason(family: number | null): string {
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/** A Chat Completions answer as an Anthropic Messages object naming `model`. */
export function openAIAnswerToAnthropic(body: Json, model: string): Json {
  const parts = openAIAnswerParts(body);
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  if (parts.text !== '') content.push({ type: 'text', text: parts.text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReason(parts.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: parts.inputTokens, output_tokens: parts.outputTokens },
  };
}

/** An Anthropic Messages answer as a Chat Completions object naming `model`. */
export function anthropicAnswerToOpenAI(body: Json, model: string): Json {
  const parts = anthropicAnswerParts(body);
  const content: Json[] = [];
  if (parts.text !== '') content.push({ type: 'text', text: parts.text });
  const blocks = Array.isArray(body.content) ? body.content : [];
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : '',
      type: 'function',
      function: {
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      },
    });
  }
  const message: Json = { role: 'assistant', content: parts.text };
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
    if (parts.text === '') message.content = null;
  }
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStopReason(parts.stopFamily), logprobs: null }],
    usage: {
      prompt_tokens: parts.inputTokens,
      completion_tokens: parts.outputTokens,
      total_tokens: parts.inputTokens + parts.outputTokens,
    },
  };
}

/** A Gemini generateContent answer as a Chat Completions object naming `model`. */
export function geminiAnswerToOpenAI(body: Json, model: string): Json {
  const parts = geminiAnswerParts(body);
  return {
    id: 'chatcmpl_gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: parts.text },
        finish_reason: openAIStopReason(parts.stopFamily),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: parts.inputTokens,
      completion_tokens: parts.outputTokens,
      total_tokens: parts.inputTokens + parts.outputTokens,
    },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object naming `model`. */
export function geminiAnswerToAnthropic(body: Json, model: string): Json {
  const parts = geminiAnswerParts(body);
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: parts.text }],
    stop_reason: anthropicStopReason(parts.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: parts.inputTokens, output_tokens: parts.outputTokens },
  };
}

// ---- Streams -------------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then a blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Every SSE frame of a stream, in order, data lines joined. */
export async function collectSse(stream: ReadableStream<Uint8Array>): Promise<SseFrame[]> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const frames: SseFrame[] = [];
  const push = (raw: string): void => {
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        push(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') push(buffer);
  } finally {
    reader.releaseLock();
  }
  return frames;
}

function jsonFrame(frame: SseFrame): Json | null {
  if (frame.data.trim() === '[DONE]') return null;
  try {
    const value: unknown = JSON.parse(frame.data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The text pieces, usage and stop of an OpenAI Chat Completions stream, in order. */
export function openAIStreamParts(frames: readonly SseFrame[]): AnswerParts {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily: number | null = null;
  for (const frame of frames) {
    const chunk = jsonFrame(frame);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') stopFamily = stopFamilyOf(choice.finish_reason);
  }
  return { text, inputTokens, outputTokens, stopFamily };
}

/** The text pieces, usage and stop of an Anthropic Messages stream, in order. */
export function anthropicStreamParts(frames: readonly SseFrame[]): AnswerParts {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily: number | null = null;
  for (const frame of frames) {
    const data = jsonFrame(frame);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') text += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
        stopFamily = stopFamilyOf(data.delta.stop_reason);
      }
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text, inputTokens, outputTokens, stopFamily };
}

/** The text pieces, usage and stop of a Gemini streamGenerateContent stream, in order. */
export function geminiStreamParts(frames: readonly SseFrame[]): AnswerParts {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily: number | null = null;
  for (const frame of frames) {
    const data = jsonFrame(frame);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') text += part.text;
      }
      if (typeof candidate.finishReason === 'string') stopFamily = stopFamilyOf(candidate.finishReason);
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, inputTokens, outputTokens, stopFamily };
}

/** Split text into the provider's own stream pieces (13 characters each) for re-emission. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** An answer's parts as an OpenAI Chat Completions SSE stream naming `model`. */
export function openAIStreamEvents(parts: AnswerParts, model: string): string {
  const finish = openAIStopReason(parts.stopFamily);
  const events: string[] = [
    sseEvent(null, {
      id: 'chatcmpl_stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
    }),
  ];
  for (const piece of splitText(parts.text)) {
    events.push(
      sseEvent(null, {
        id: 'chatcmpl_stream',
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
      }),
    );
  }
  events.push(
    sseEvent(null, {
      id: 'chatcmpl_stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: finish }],
    }),
    sseEvent(null, {
      id: 'chatcmpl_stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: parts.inputTokens,
        completion_tokens: parts.outputTokens,
        total_tokens: parts.inputTokens + parts.outputTokens,
      },
    }),
    'data: [DONE]\n\n',
  );
  return events.join('');
}

/** An answer's parts as an Anthropic Messages SSE stream naming `model`. */
export function anthropicStreamEvents(parts: AnswerParts, model: string): string {
  const events: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: parts.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(parts.text)) {
    events.push(
      sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }),
    );
  }
  events.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopReason(parts.stopFamily), stop_sequence: null },
      usage: { output_tokens: parts.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events.join('');
}
