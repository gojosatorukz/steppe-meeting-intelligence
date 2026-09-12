import { useEffect, useRef, useState } from "react";
import {
  AudioLines,
  FileText,
  UploadCloud,
  LoaderCircle,
  LockKeyhole,
  AlertCircle,
} from "lucide-react";
import {
  processAudio,
  processText,
  PROTOCOL_LANGUAGES,
  type ProtocolLanguage,
  type Result,
} from "../api";
import { Modal } from "./Modal";
export function Upload({
  onClose,
  onComplete,
}: {
  onClose: () => void;
  onComplete: (r: Result, title: string, file?: File) => Promise<void>;
}) {
  const [mode, setMode] = useState<"audio" | "text">("audio"),
    [file, setFile] = useState<File>(),
    [text, setText] = useState(""),
    [title, setTitle] = useState(""),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [seconds, setSeconds] = useState(0),
    [drag, setDrag] = useState(false),
    [lang, setLang] = useState<ProtocolLanguage>("ru");
  const input = useRef<HTMLInputElement>(null),
    controller = useRef<AbortController | undefined>(undefined);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    if (!busy) return;
    const start = Date.now();
    const timer = setInterval(
      () => setSeconds(Math.floor((Date.now() - start) / 1000)),
      1000,
    );
    return () => clearInterval(timer);
  }, [busy]);
  function choose(f?: File) {
    if (!f) return;
    setError("");
    if (!/\.(mp3|wav|m4a|ogg|flac|mp4|webm)$/i.test(f.name)) {
      setError(
        "Выберите аудио в формате MP3, WAV, M4A, OGG, FLAC, MP4 или WebM.",
      );
      return;
    }
    if (!f.size || f.size > 200 * 1024 * 1024) {
      setError("Размер файла должен быть от 1 байта до 200 МБ.");
      return;
    }
    setFile(f);
    if (!title) setTitle(f.name.replace(/\.[^.]+$/, ""));
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setError("");
    if (mode === "audio" && !file) {
      setError("Добавьте запись встречи.");
      return;
    }
    if (mode === "text" && !text.trim()) {
      setError("Вставьте текст встречи.");
      return;
    }
    const c = new AbortController();
    controller.current = c;
    setBusy(true);
    setSeconds(0);
    try {
      const r =
        mode === "audio"
          ? await processAudio(file!, c.signal, lang)
          : await processText(text.trim(), c.signal, lang);
      await onComplete(r, title.trim(), mode === "audio" ? file : undefined);
    } catch (e) {
      if (!c.signal.aborted)
        setError(e instanceof Error ? e.message : "Ошибка обработки.");
    } finally {
      if (controller.current === c) setBusy(false);
    }
  }
  return (
    <Modal
      title="Новая встреча"
      onClose={() => {
        if (!busy) onClose();
      }}
    >
      <p className="modal-description">
        Добавьте запись или готовую расшифровку.
      </p>
      {busy ? (
        <section className="processing">
          <span className="processing-icon">
            <AudioLines size={36} />
          </span>
          <h3>
            {mode === "audio"
              ? "Распознаём речь и формируем протокол"
              : "Формируем протокол по тексту"}
          </h3>
          <p>
            Локальные модели обрабатывают встречу.
            <br />
            Прошло {seconds} сек.
          </p>
          <div className="processing-live">
            <LoaderCircle size={18} className="spin" />
            Ожидаем ответ сервера
          </div>
          <p>Можно отключить интернет. Оставьте окно открытым до завершения.</p>
          <button
            className="button secondary"
            onClick={() => {
              controller.current?.abort();
              setBusy(false);
              setError(
                "Ожидание прекращено. Обработка на сервере может продолжаться. Закройте окно и обновите список встреч перед повторной отправкой.",
              );
            }}
          >
            Прекратить ожидание
          </button>
        </section>
      ) : (
        <form onSubmit={submit}>
          <div
            className="segmented"
            role="tablist"
            aria-label="Источник встречи"
          >
            <button
              type="button"
              role="tab"
              aria-selected={mode === "audio"}
              onClick={() => setMode("audio")}
            >
              <AudioLines size={17} />
              Аудиозапись
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "text"}
              onClick={() => setMode("text")}
            >
              <FileText size={17} />
              Готовый текст
            </button>
          </div>
          <label className="form-field">
            Название
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Название встречи"
              maxLength={180}
            />
          </label>
          <label className="form-field">
            Язык протокола
            <select
              value={lang}
              disabled={busy}
              onChange={(e) => setLang(e.target.value as ProtocolLanguage)}
            >
              {PROTOCOL_LANGUAGES.map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>
          </label>
          {mode === "audio" ? (
            <>
              <input
                ref={input}
                className="sr-only"
                type="file"
                accept=".mp3,.wav,.m4a,.ogg,.flac,.mp4,.webm"
                onChange={(e) => choose(e.target.files?.[0])}
                aria-label="Выбрать аудиофайл"
              />
              <button
                type="button"
                className={`file-drop ${drag ? "dragging" : ""}`}
                onClick={() => input.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDrag(false);
                  choose(e.dataTransfer.files[0]);
                }}
              >
                <UploadCloud size={32} />
                <strong>
                  {file ? file.name : "Перетащите аудиозапись сюда"}
                </strong>
                <span>
                  {file
                    ? `${(file.size / 1024 / 1024).toFixed(1)} МБ · нажмите, чтобы заменить`
                    : "или выберите файл на устройстве"}
                </span>
                <small>MP3 · WAV · M4A · OGG · FLAC · MP4 · WebM</small>
              </button>
            </>
          ) : (
            <label className="form-field">
              Текст встречи
              <textarea
                rows={8}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Вставьте расшифровку встречи…"
                maxLength={200000}
              />
              <span className="field-hint">
                Русский, казахский, английский и смешанная речь.
              </span>
            </label>
          )}
          {error && (
            <div className="error-banner" role="alert">
              <AlertCircle size={18} />
              {error}
            </div>
          )}
          <div className="privacy-inline">
            <LockKeyhole size={15} />
            Только ваш локальный сервер
          </div>
          <button className="button primary full" type="submit">
            <AudioLines size={17} />
            Создать протокол
          </button>
        </form>
      )}
    </Modal>
  );
}
