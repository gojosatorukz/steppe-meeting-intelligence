import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  AudioLines,
  Mic,
  Square,
  LoaderCircle,
  ShieldCheck,
  Volume2,
} from "lucide-react";
import {
  liveStatus,
  liveStart,
  liveStop,
  PROTOCOL_LANGUAGES,
  type ProtocolLanguage,
  micTest,
  result,
  duration,
  type LiveEvent,
  type LiveStatus,
  type Result,
} from "../api";
const states: Record<string, string> = {
  idle: "Готовы к записи",
  running: "Идёт запись",
  finalizing: "Формируем протокол",
  done: "Запись завершена",
  error: "Ошибка записи",
};
export function Live({
  active,
  onComplete,
  onState,
}: {
  active: boolean;
  onComplete: (r: Result, title: string) => Promise<void>;
  onState: (s: string) => void;
}) {
  const [status, setStatus] = useState<LiveStatus | null>(null),
    [state, setState] = useState("idle"),
    [device, setDevice] = useState(""),
    [count, setCount] = useState(""),
    [names, setNames] = useState(""),
    [title, setTitle] = useState(""),
    [events, setEvents] = useState<LiveEvent[]>([]),
    [level, setLevel] = useState(0),
    [seconds, setSeconds] = useState(0),
    [pending, setPending] = useState(0),
    [connected, setConnected] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [mic, setMic] = useState(""),
    [lastId, setLastId] = useState<string | null>(null),
    [lang, setLang] = useState<ProtocolLanguage>("ru");
  const callback = useRef(onComplete);
  callback.current = onComplete;
  const resultIds = useRef(new Set<string>());
  const finishing = useRef(false);
  const titleRef = useRef(title);
  titleRef.current = title;
  const tape = useRef<HTMLDivElement>(null),
    follow = useRef(true);
  useEffect(() => onState(state), [state, onState]);
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    liveStatus()
      .then((s) => {
        if (cancelled) return;
        setStatus(s);
        setState(s.state);
        setSeconds(s.audio_seconds);
        setPending(s.pending);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [active]);
  useEffect(() => {
    if (!active) return;
    let disposed = false;
    let socket: WebSocket;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      socket = new WebSocket(
        `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/live/ws`,
      );
      socket.onopen = () => {
        setConnected(true);
        setEvents([]);
      };
      socket.onmessage = (e) => {
        let m: LiveEvent;
        try {
          m = JSON.parse(e.data);
        } catch {
          return;
        }
        if (m.type === "status" && m.state) setState(m.state);
        if (m.type === "level") {
          setLevel(Math.min(100, Math.max(0, (m.rms || 0) * 600)));
          setSeconds(m.seconds || 0);
        }
        if (m.type === "queue") setPending(m.pending || 0);
        if (m.type === "segment")
          setEvents((prev) => [
            ...prev.filter(
              (x) =>
                !(x.type === "pending" && x.utterance_id === m.utterance_id),
            ),
            m,
          ]);
        if (m.type === "pending") setEvents((prev) => [...prev, m]);
        if (m.type === "drop")
          setEvents((prev) =>
            prev.filter((x) => x.utterance_id !== m.utterance_id),
          );
        if (m.type === "done" && m.id) {
          setState("done");
          setLastId(m.id);
        }
      };
      socket.onerror = () => setConnected(false);
      socket.onclose = () => {
        setConnected(false);
        if (!disposed) retry = setTimeout(connect, 2500);
      };
    };
    connect();
    return () => {
      disposed = true;
      clearTimeout(retry);
      socket?.close();
    };
  }, [active]);
  useEffect(() => {
    if (follow.current && tape.current)
      tape.current.scrollTop = tape.current.scrollHeight;
  }, [events]);
  async function start() {
    setError("");
    setBusy(true);
    try {
      const r = await liveStart(
        device === "" ? null : Number(device),
        count === "" ? null : Number(count),
        names
          .split(",")
          .map((x) => x.trim())
          .filter(Boolean),
        lang,
      );
      setEvents([]);
      setSeconds(0);
      setPending(0);
      setMic(r.device);
      setState("running");
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Не удалось включить микрофон.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function stop() {
    setError("");
    setBusy(true);
    finishing.current = true;
    setState("finalizing");
    try {
      const r = await liveStop();
      resultIds.current.add(r.id);
      setState("done");
      await callback.current(r, title);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Не удалось завершить встречу.",
      );
      try {
        const s = await liveStatus();
        setState(s.state);
      } catch {
        setState("error");
      }
    } finally {
      finishing.current = false;
      setBusy(false);
    }
  }
  async function testMic() {
    setBusy(true);
    setError("");
    setMic("Проверяем микрофон: запись 2 секунды…");
    try {
      const r = await micTest(device === "" ? null : Number(device));
      setMic(
        `${r.device} · ${r.ok ? "Микрофон слышно" : "Слишком тихо"} · уровень ${r.rms.toFixed(4)}`,
      );
      setLevel(Math.min(100, r.rms * 600));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Микрофон недоступен.");
      setMic("");
    } finally {
      setBusy(false);
    }
  }
  const recording = state === "running" || state === "finalizing";
  return (
    <div hidden={!active}>
      <div className="page-heading">
        <div>
          <div className="eyebrow">РАЗГОВОР СТАНОВИТСЯ ТЕКСТОМ</div>
          <h1>Живая встреча</h1>
          <p>
            Запись с микрофона сервера, расшифровка и протокол в одном месте.
          </p>
        </div>
        <span className={`pill ${recording ? "recording-pill" : "local"}`}>
          <span className={recording ? "record-dot" : ""} />
          {states[state] || state}
        </span>
      </div>
      <div className="live-grid">
        <section className="panel">
          <div className="section-heading">
            <h2>
              <Mic size={19} />
              Подготовка встречи
            </h2>
          </div>
          <label className="form-field">
            Название
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={recording}
              placeholder="Название встречи"
            />
          </label>
          <label className="form-field">
            Участники
            <input
              value={names}
              onChange={(e) => setNames(e.target.value)}
              disabled={recording}
              placeholder="Имена через запятую"
            />
            <span className="field-hint">
              Помогает распознавать имена. Голоса будут обозначены «Спикер 1»,
              «Спикер 2».
            </span>
          </label>
          <div className="form-columns">
            <label className="form-field">
              Число спикеров
              <select
                value={count}
                disabled={recording}
                onChange={(e) => setCount(e.target.value)}
              >
                <option value="">Автоматически</option>
                {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label className="form-field">
              Язык протокола
              <select
                value={lang}
                disabled={recording}
                onChange={(e) => setLang(e.target.value as ProtocolLanguage)}
              >
                {PROTOCOL_LANGUAGES.map((l) => (
                  <option key={l.value} value={l.value}>
                    {l.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="form-field">
              Микрофон
              <input
                type="number"
                min="0"
                step="1"
                disabled={recording}
                value={device}
                placeholder="По умолчанию"
                onChange={(e) => setDevice(e.target.value)}
              />
            </label>
          </div>
          <div className="mic-meter">
            <span>
              <Volume2 size={16} />
              Уровень микрофона
            </span>
            <div className="level-track">
              <i style={{ width: level + "%" }} />
            </div>
          </div>
          {mic && <p className="field-hint mic-message">{mic}</p>}
          <div className="live-controls">
            {recording ? (
              <button
                className="button danger"
                onClick={stop}
                disabled={busy || state === "finalizing"}
              >
                {busy ? (
                  <LoaderCircle size={16} className="spin" />
                ) : (
                  <Square size={16} />
                )}
                Завершить и получить протокол
              </button>
            ) : (
              <button
                className="button primary"
                onClick={start}
                disabled={
                  busy ||
                  !connected ||
                  status?.audio_available === false ||
                  status?.vad_available === false
                }
              >
                {busy ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Mic size={16} />
                )}
                Начать запись
              </button>
            )}
            <button
              className="button secondary"
              disabled={recording || busy}
              onClick={testMic}
            >
              Проверить микрофон
            </button>
          </div>
          {status && (!status.audio_available || !status.vad_available) && (
            <div className="error-banner">
              <AlertCircle size={17} />
              {!status.audio_available
                ? "Аудиозахват недоступен на сервере."
                : "На сервере не найден VAD-модуль. Добавьте локальную модель для живой записи."}
            </div>
          )}
          {error && (
            <div className="error-banner" role="alert">
              <AlertCircle size={17} />
              {error}
            </div>
          )}
          <div className="privacy-inline">
            <ShieldCheck size={15} />
            Используется микрофон компьютера с backend
          </div>
        </section>
        <section className="panel live-transcript">
          <div className="section-heading">
            <h2>
              <AudioLines size={18} />
              Расшифровка
            </h2>
            <span
              className={`connection-label ${connected ? "connected" : ""}`}
            >
              {connected ? "Поток подключён" : "Нет связи с потоком"}
            </span>
          </div>
          <div className="live-metrics">
            <div>
              <span>Записано</span>
              <strong>{duration(seconds)}</strong>
            </div>
            <div>
              <span>Реплик</span>
              <strong>
                {events.filter((e) => e.type === "segment").length}
              </strong>
            </div>
            <div>
              <span>В очереди</span>
              <strong>{pending}</strong>
            </div>
          </div>
          {state === "done" && lastId && (
            <button
              className="button secondary full"
              onClick={() =>
                result(lastId)
                  .then((r) => callback.current(r, title))
                  .catch((e) => setError(e.message))
              }
            >
              Открыть готовый протокол
            </button>
          )}
          <div
            className="live-tape"
            ref={tape}
            onScroll={() => {
              const t = tape.current!;
              follow.current =
                t.scrollHeight - t.scrollTop - t.clientHeight < 60;
            }}
          >
            {events.length ? (
              events.map((e, i) => (
                <div
                  key={`${e.utterance_id}-${e.start}-${i}`}
                  className={`live-line ${e.type === "pending" ? "pending" : ""}`}
                >
                  <span>{duration(e.start || 0)}</span>
                  <div>
                    {e.type === "pending" ? (
                      <span className="pending-label">
                        <LoaderCircle className="spin" size={13} />
                        Распознаём реплику…
                      </span>
                    ) : (
                      <>
                        <span className="language-tag">
                          {e.language?.toUpperCase() || "Речь"}
                        </span>
                        <p>{e.text}</p>
                      </>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="empty live-empty">
                <AudioLines size={32} />
                <h3>
                  {recording ? "Слушаем встречу" : "Здесь появится ваша речь"}
                </h3>
                <p>
                  {recording
                    ? "Реплики появятся после пауз в разговоре."
                    : "Начните запись — текст будет появляться по мере разговора."}
                </p>
              </div>
            )}
          </div>
          {state === "finalizing" && (
            <div className="live-finalizing">
              <LoaderCircle size={17} className="spin" />
              Завершаем распознавание, определяем спикеров и составляем
              протокол.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
