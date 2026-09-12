export type ActionItem = {
  owner: string;
  task: string;
  due: string;
  priority: string;
};
export type Segment = {
  start: number;
  end: number;
  text: string;
  language?: string;
  speaker?: string | null;
  utterance_id?: number;
};
export type Protocol = {
  summary: string[];
  decisions: string[];
  topics: { title: string; points: string[] }[];
  open_questions: string[];
  action_items: ActionItem[];
  risks: string[];
};
export type Result = {
  id: string;
  filename?: string;
  transcript: string;
  language: string;
  languages?: Record<string, number>;
  participants?: string[];
  audio_seconds: number;
  segments: Segment[];
  speakers?: Segment[];
  protocol: Protocol;
  protocol_error?: string | null;
  timing: Record<string, number | null>;
};
export type RecordItem = { result: Result; title: string; receivedAt?: string };
export type Health = {
  ok: boolean;
  offline: boolean;
  warm?: boolean;
  // T5 renamed the STT fields; `whisper_size` is the pre-T5 spelling and is
  // kept so an older backend still renders instead of showing "Нет данных".
  stt_models?: { turbo?: string; kk?: string; kk_available?: boolean };
  stt_device?: string;
  whisper_size?: string;
  diarization?: { available?: boolean; threshold?: number };
  offline_guard?: { enabled?: boolean; blocked_external_connections?: number };
  llm_model: string | null;
  llm_error?: string | null;
  ollama_running: boolean;
  ollama_models: string[];
  llm_ready: boolean;
  results_stored: number;
};
export type LiveStatus = {
  state: string;
  device: string | null;
  pending: number;
  segments: number;
  audio_seconds: number;
  audio_available: boolean;
  vad_available: boolean;
  diarization_available: boolean;
};
export type LiveEvent = {
  type: string;
  utterance_id?: number;
  start?: number;
  end?: number;
  text?: string;
  language?: string;
  speaker?: string | null;
  pending?: number;
  rms?: number;
  peak?: number;
  seconds?: number;
  state?: string;
  id?: string;
};
const obj = (x: unknown): x is Record<string, unknown> =>
  !!x && typeof x === "object" && !Array.isArray(x);
const strings = (x: unknown): x is string[] =>
  Array.isArray(x) && x.every((v) => typeof v === "string");
export function parseResult(x: unknown): Result {
  if (
    !obj(x) ||
    typeof x.id !== "string" ||
    typeof x.transcript !== "string" ||
    !obj(x.protocol) ||
    typeof x.language !== "string" ||
    typeof x.audio_seconds !== "number" ||
    !Number.isFinite(x.audio_seconds) ||
    x.audio_seconds < 0
  )
    throw new Error("Сервер вернул неполный результат обработки.");
  const p = x.protocol;
  for (const k of ["summary", "decisions", "open_questions", "risks"])
    if (p[k] !== undefined && !strings(p[k]))
      throw new Error("Некорректный раздел протокола: " + k);
  if (
    p.topics !== undefined &&
    (!Array.isArray(p.topics) ||
      !p.topics.every(
        (t) => obj(t) && typeof t.title === "string" && strings(t.points),
      ))
  )
    throw new Error("Некорректные темы протокола.");
  if (
    p.action_items !== undefined &&
    (!Array.isArray(p.action_items) ||
      !p.action_items.every(
        (t) =>
          obj(t) &&
          ["owner", "task", "due", "priority"].every(
            (k) => typeof t[k] === "string",
          ),
      ))
  )
    throw new Error("Некорректная таблица поручений.");
  const segments = (value: unknown): Segment[] =>
    Array.isArray(value)
      ? (value.filter(
          (s) =>
            obj(s) &&
            typeof s.text === "string" &&
            typeof s.start === "number" &&
            typeof s.end === "number" &&
            Number.isFinite(s.start) &&
            Number.isFinite(s.end) &&
            s.end >= s.start,
        ) as Segment[])
      : [];
  return {
    ...x,
    segments: segments(x.segments),
    speakers: segments(x.speakers),
    protocol: {
      summary: [],
      decisions: [],
      topics: [],
      open_questions: [],
      action_items: [],
      risks: [],
      ...p,
    },
    timing: obj(x.timing) ? x.timing : {},
  } as unknown as Result;
}
export async function request<T>(
  url: string,
  options: RequestInit = {},
  timeout = 15000,
): Promise<T> {
  const signal = options.signal
    ? AbortSignal.any([options.signal, AbortSignal.timeout(timeout)])
    : AbortSignal.timeout(timeout);
  let response: Response;
  try {
    response = await fetch(url, { ...options, signal, redirect: "error" });
  } catch (e) {
    if (options.signal?.aborted)
      throw new DOMException("Ожидание остановлено", "AbortError");
    if (signal.aborted)
      throw new Error(
        "Время ожидания истекло. Сервер мог продолжить обработку — обновите список встреч перед повторной отправкой.",
      );
    throw new Error(
      "Нет связи с локальным сервером. Убедитесь, что FastAPI запущен на порту 8000.",
    );
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error("Сервер вернул ответ, который не удалось прочитать.");
  }
  if (!response.ok) {
    const detail = obj(body) ? body.detail : null;
    throw new Error(
      typeof detail === "string"
        ? detail
        : `Ошибка сервера ${response.status}.`,
    );
  }
  return body as T;
}
export async function health(signal?: AbortSignal) {
  return request<Health>("/health", { signal }, 8000);
}
export async function results(signal?: AbortSignal) {
  const data = await request<unknown>("/results", { signal });
  if (
    !Array.isArray(data) ||
    !data.every((r) => obj(r) && typeof r.id === "string")
  )
    throw new Error("Неверный формат истории встреч.");
  return data as { id: string; filename?: string }[];
}
export async function result(id: string, signal?: AbortSignal) {
  return parseResult(
    await request("/result/" + encodeURIComponent(id), { signal }),
  );
}
export type ChatSource = {
  start: number;
  end: number;
  speaker: string;
  text: string;
};
export type ChatAnswer = {
  answer: string;
  sources: ChatSource[];
  language?: string;
};
/** Ask a question about one stored meeting. Answered locally, from its own
 *  transcript — the timeout is generous because a 4B model on a shared card
 *  takes tens of seconds for a first answer. */
export async function askMeeting(
  id: string,
  question: string,
  language: ProtocolLanguage = "auto",
  signal?: AbortSignal,
) {
  const body = await request<ChatAnswer>(
    `/result/${encodeURIComponent(id)}/chat`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, language }),
      signal,
    },
    5 * 60 * 1000,
  );
  return {
    answer: typeof body?.answer === "string" ? body.answer : "",
    sources: Array.isArray(body?.sources) ? body.sources : [],
    language: body?.language,
  } satisfies ChatAnswer;
}
export type ProtocolLanguage = "ru" | "kk" | "en" | "auto";
export const PROTOCOL_LANGUAGES: { value: ProtocolLanguage; label: string }[] = [
  { value: "ru", label: "Русский" },
  { value: "kk", label: "Қазақша" },
  { value: "en", label: "English" },
  { value: "auto", label: "Авто" },
];
export async function processAudio(
  file: File,
  signal: AbortSignal,
  protocol_language: ProtocolLanguage = "ru",
) {
  const body = new FormData();
  body.set("file", file);
  body.set("protocol_language", protocol_language);
  return parseResult(
    await request("/process", { method: "POST", body, signal }, 30 * 60 * 1000),
  );
}
export async function processText(
  transcript: string,
  signal: AbortSignal,
  protocol_language: ProtocolLanguage = "ru",
) {
  return parseResult(
    await request(
      "/process-text",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript, protocol_language }),
        signal,
      },
      30 * 60 * 1000,
    ),
  );
}
export const liveStatus = () => request<LiveStatus>("/live/status");
export const liveStart = (
  device: number | null,
  num_speakers: number | null,
  participants: string[],
  protocol_language: ProtocolLanguage = "ru",
) =>
  request<{ ok: boolean; device: string }>(
    "/live/start",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device,
        num_speakers,
        participants,
        protocol_language,
      }),
    },
    120000,
  );
export const liveStop = async () =>
  parseResult(await request("/live/stop", { method: "POST" }, 30 * 60 * 1000));
export const micTest = (device: number | null) =>
  request<{ ok: boolean; rms: number; peak: number; device: string }>(
    "/mic-test?seconds=2" + (device === null ? "" : "&device=" + device),
    {},
    15000,
  );
export const titleOf = (r: Result) =>
  r.filename === "(live)"
    ? "Живая встреча"
    : r.filename === "(text input)"
      ? "Текстовая встреча"
      : r.filename || "Встреча";
export const duration = (s: number) =>
  `${Math.floor(s / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(s % 60)
    .toString()
    .padStart(2, "0")}`;
export const languageName = (l: string) =>
  ({
    ru: "Русский",
    kk: "Қазақша",
    en: "English",
    mixed: "Смешанная речь",
    "n/a": "Текст",
    unknown: "Не определён",
  })[l] || l.toUpperCase();
export const priorityClass = (s: string) =>
  /высок|high/i.test(s)
    ? "high"
    : /низк|low/i.test(s)
      ? "low"
      : /сред|medium/i.test(s)
        ? "medium"
        : "unknown";
export const sttLabel = (h?: Health | null) => {
  const name = h?.stt_models?.turbo || h?.whisper_size;
  if (!name) return "Нет данных";
  return h?.stt_device ? `${name} · ${h.stt_device}` : name;
};
