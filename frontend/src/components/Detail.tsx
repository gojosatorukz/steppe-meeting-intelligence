import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  AudioLines,
  CalendarDays,
  CheckCheck,
  ChevronDown,
  Clock3,
  Download,
  FileText,
  Lightbulb,
  ListTodo,
  Search,
  ShieldCheck,
  TriangleAlert,
  Users,
  MessageCircle,
  LoaderCircle,
} from "lucide-react";
import {
  askMeeting,
  duration,
  languageName,
  priorityClass,
  type ActionItem,
  type ChatSource,
  type RecordItem,
} from "../api";
import { readAudio } from "../storage";
export function TasksTable({ items }: { items: ActionItem[] }) {
  return items.length ? (
    <div className="table-scroll">
      <table className="tasks-table">
        <thead>
          <tr>
            <th>Поручение</th>
            <th>Ответственный</th>
            <th>Срок</th>
            <th>Приоритет</th>
          </tr>
        </thead>
        <tbody>
          {items.map((t, i) => (
            <tr key={i}>
              <td className="task-description">{t.task}</td>
              <td>
                {t.owner || <span className="missing-value">Не назначен</span>}
              </td>
              <td>
                {t.due || <span className="missing-value">Не указан</span>}
              </td>
              <td>
                <span className={`priority ${priorityClass(t.priority)}`}>
                  {t.priority || "Не указан"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ) : (
    <div className="empty">
      <ListTodo size={27} />
      <h3>Поручения не выделены</h3>
      <p>В этой встрече модель не нашла конкретных задач.</p>
    </div>
  );
}
function save(content: Blob, name: string) {
  const url = URL.createObjectURL(content);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
export function csv(items: ActionItem[]) {
  const cell = (s: string) =>
    '"' + (/^[\s]*[=+@-]/.test(s) ? "'" + s : s).replaceAll('"', '""') + '"';
  return (
    "\uFEFF" +
    [
      ["Ответственный", "Задача", "Срок", "Приоритет"],
      ...items.map((t) => [t.owner, t.task, t.due, t.priority]),
    ]
      .map((row) => row.map(cell).join(","))
      .join("\r\n")
  );
}
export function Detail({
  record,
  onBack,
  notify,
}: {
  record: RecordItem;
  onBack: () => void;
  notify: (s: string) => void;
}) {
  const r = record.result,
    p = r.protocol;
  const [tab, setTab] = useState("protocol"),
    [query, setQuery] = useState(""),
    [audio, setAudio] = useState<string>(),
    [menu, setMenu] = useState(false),
    [downloading, setDownloading] = useState(false),
    [question, setQuestion] = useState(""),
    [asking, setAsking] = useState(false),
    [chatError, setChatError] = useState(""),
    [chat, setChat] = useState<
      { question: string; answer: string; sources: ChatSource[] }[]
    >([]);
  const audioRef = useRef<HTMLAudioElement>(null),
    menuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let dead = false,
      url: string | undefined;
    readAudio(r.id)
      .then((blob) => {
        if (dead) return;
        if (blob) {
          url = URL.createObjectURL(blob);
          setAudio(url);
        } else if (r.filename === "(live)")
          setAudio(`/result/${encodeURIComponent(r.id)}/audio`);
      })
      .catch(() => {});
    return () => {
      dead = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [r.id]);
  useEffect(() => {
    if (!menu) return;
    const close = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenu(false);
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenu(false);
    };
    document.addEventListener("click", close);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("click", close);
      document.removeEventListener("keydown", key);
    };
  }, [menu]);
  async function ask(e: React.FormEvent) {
    e.preventDefault();
    const asked = question.trim();
    if (!asked || asking) return;
    setAsking(true);
    setChatError("");
    try {
      const reply = await askMeeting(r.id, asked);
      setChat((turns) => [
        ...turns,
        { question: asked, answer: reply.answer, sources: reply.sources },
      ]);
      setQuestion("");
    } catch (err) {
      setChatError(
        err instanceof Error ? err.message : "Не удалось получить ответ.",
      );
    } finally {
      setAsking(false);
    }
  }
  async function download(format: "json" | "csv" | "pdf" | "ics") {
    setMenu(false);
    setDownloading(true);
    try {
      let blob: Blob;
      if (format === "json")
        blob = new Blob([JSON.stringify(r, null, 2)], {
          type: "application/json;charset=utf-8",
        });
      else if (format === "csv")
        blob = new Blob([csv(p.action_items)], {
          type: "text/csv;charset=utf-8",
        });
      else {
        const response = await fetch(
          `/result/${encodeURIComponent(r.id)}/export.${format}`,
          { signal: AbortSignal.timeout(60000) },
        );
        if (!response.ok) {
          if (response.status === 404)
            throw new Error(
              "Этой встречи уже нет в памяти сервера. JSON и CSV доступны из локальной копии.",
            );
          let detail = `Не удалось сформировать ${format.toUpperCase()}.`;
          try {
            detail = (await response.json()).detail || detail;
          } catch {}
          throw new Error(detail);
        }
        blob = await response.blob();
      }
      save(
        blob,
        `${record.title.replace(/[^\p{L}\p{N} _-]/gu, "").slice(0, 80) || "protocol"}.${format}`,
      );
      notify(`${format.toUpperCase()} сохранён.`);
    } catch (e) {
      notify(e instanceof Error ? e.message : "Не удалось скачать файл.");
    } finally {
      setDownloading(false);
    }
  }
  const blocks = r.speakers?.length ? r.speakers : r.segments;
  const speakers = [
    ...new Set(blocks.map((s) => s.speaker).filter(Boolean)),
  ] as string[];
  const anyProtocol =
    p.summary.length ||
    p.decisions.length ||
    p.action_items.length ||
    p.topics.length;
  const list = (items: string[]) =>
    items.length ? (
      <div className="real-list">
        {items.map((s, i) => (
          <div key={i}>
            <span className="list-bullet" />
            <p>{s}</p>
          </div>
        ))}
      </div>
    ) : (
      <p className="empty-line">Не зафиксированы в этой встрече.</p>
    );
  const secs = (n: number | null | undefined) =>
    n == null ? "—" : `${n.toFixed(1)} с`;
  return (
    <>
      <div className="detail-back">
        <button onClick={onBack}>
          <ArrowLeft size={16} />
          Все встречи
        </button>
        <span className="pill local">
          <ShieldCheck size={13} />
          На вашем устройстве
        </span>
      </div>
      <div className="page-heading detail-heading">
        <div>
          <div className="eyebrow">ПРОТОКОЛ ВСТРЕЧИ</div>
          <h1>{record.title}</h1>
          <div className="detail-meta">
            <span>
              <Clock3 size={15} />
              {r.audio_seconds
                ? duration(r.audio_seconds)
                : "Текстовая встреча"}
            </span>
            <span className="language-tag">{languageName(r.language)}</span>
            {speakers.length > 0 && (
              <span>
                <Users size={15} />
                {speakers.length} спикера
              </span>
            )}
            {record.receivedAt && (
              <span>
                <CalendarDays size={15} />
                Сохранено{" "}
                {new Date(record.receivedAt).toLocaleDateString("ru-RU")}
              </span>
            )}
          </div>
        </div>
        <div className="export-dropdown" ref={menuRef}>
          <button
            className="button primary"
            onClick={() => setMenu(!menu)}
            aria-expanded={menu}
            disabled={downloading}
          >
            {downloading ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <Download size={17} />
            )}
            Экспорт
            <ChevronDown size={14} />
          </button>
          {menu && (
            <div className="dropdown-menu">
              {(["pdf", "csv", "json", "ics"] as const).map((f) => (
                <button key={f} onClick={() => download(f)}>
                  <FileText size={17} />
                  <span>
                    {f.toUpperCase()}
                    <small>
                      {
                        {
                          pdf: "Оформленный протокол",
                          csv: "Таблица поручений",
                          json: "Полный результат обработки",
                          ics: "Поручения в календарь",
                        }[f]
                      }
                    </small>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      {!anyProtocol && (
        <div className="error-banner" role="alert">
          <TriangleAlert size={19} />
          Сервер вернул расшифровку, но не сформировал содержательный протокол.
          Проверьте локальную LLM. Текст встречи доступен во вкладке
          «Расшифровка».
        </div>
      )}
      <div className="meeting-body real-detail">
        <div className="meeting-primary">
          <div
            className="detail-tabs"
            role="tablist"
            aria-label="Материалы встречи"
          >
            {[
              ["protocol", "Протокол", FileText],
              ["tasks", "Поручения", ListTodo],
              ["transcript", "Расшифровка", AudioLines],
            ].map(([v, label, Icon]) => {
              const I = Icon as typeof FileText;
              return (
                <button
                  key={v as string}
                  role="tab"
                  aria-selected={tab === v}
                  className={tab === v ? "selected" : ""}
                  onClick={() => setTab(v as string)}
                >
                  <I size={16} />
                  {label as string}
                  {v === "tasks" && (
                    <span className="count">{p.action_items.length}</span>
                  )}
                </button>
              );
            })}
          </div>
          {r.protocol_error && (
            <div className="error-banner" role="alert">
              <TriangleAlert size={18} />
              Протокол не сформирован: {r.protocol_error}. Расшифровка ниже —
              настоящая.
            </div>
          )}
          {tab === "protocol" && (
            <>
              <section className="summary-panel">
                <div className="section-heading">
                  <h2>
                    <FileText size={17} />
                    Кратко о главном
                  </h2>
                  <span className="summary-label">EXECUTIVE SUMMARY</span>
                </div>
                {p.summary.length ? (
                  p.summary.map((s, i) => (
                    <p key={i} className="summary-sentence">
                      {s}
                    </p>
                  ))
                ) : (
                  <p>Краткое содержание не сформировано.</p>
                )}
                <div className="summary-bottom">
                  <span>На основе вашей встречи</span>
                  <button onClick={() => setTab("transcript")}>
                    Открыть расшифровку
                  </button>
                </div>
              </section>
              <section className="panel">
                <div className="section-heading">
                  <h2>
                    <CheckCheck size={18} />
                    Принятые решения{" "}
                    <span className="count">{p.decisions.length}</span>
                  </h2>
                </div>
                {list(p.decisions)}
              </section>
              <section className="panel">
                <div className="section-heading">
                  <h2>
                    <Lightbulb size={18} />
                    Темы и тезисы
                  </h2>
                </div>
                {p.topics.length ? (
                  <div className="topics">
                    {p.topics.map((t, i) => (
                      <div key={i}>
                        <span>{String(i + 1).padStart(2, "0")}</span>
                        <div>
                          <h3>{t.title}</h3>
                          <ul>
                            {t.points.map((s, j) => (
                              <li key={j}>{s}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="empty-line">Отдельные темы не выделены.</p>
                )}
              </section>
              <div className="protocol-bottom">
                <section className="panel">
                  <div className="section-heading">
                    <h2>
                      <MessageCircle size={17} />
                      Открытые вопросы
                    </h2>
                  </div>
                  {list(p.open_questions)}
                </section>
                <section className="panel risk-panel">
                  <div className="section-heading">
                    <h2>
                      <TriangleAlert size={17} />
                      Риски и блокеры
                    </h2>
                  </div>
                  {list(p.risks)}
                </section>
              </div>
              <section className="panel meeting-chat">
                <div className="section-heading">
                  <h2>
                    <MessageCircle size={17} />
                    Вопрос по встрече
                  </h2>
                </div>
                <p className="section-description">
                  Отвечает локальная модель и только по расшифровке этой
                  встречи. Под ответом — фрагменты, на которые она опиралась.
                </p>
                {chat.length > 0 && (
                  <div className="chat-thread">
                    {chat.map((turn, i) => (
                      <div className="chat-turn" key={i}>
                        <p className="chat-question">{turn.question}</p>
                        <p className="chat-answer">{turn.answer}</p>
                        {turn.sources.length > 0 && (
                          <div className="chat-sources">
                            {turn.sources.map((s, j) => (
                              <span className="chat-chip" key={j} title={s.text}>
                                {duration(s.start)} · {s.speaker || "фрагмент"}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                <form className="chat-form" onSubmit={ask}>
                  <input
                    aria-label="Вопрос по встрече"
                    placeholder="Например: что решили по бюджету?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    disabled={asking}
                  />
                  <button
                    className="button primary"
                    type="submit"
                    disabled={asking || !question.trim()}
                  >
                    {asking ? (
                      <LoaderCircle className="spin" size={17} />
                    ) : (
                      <MessageCircle size={17} />
                    )}
                    {asking ? "Отвечаю…" : "Спросить"}
                  </button>
                </form>
                {chatError && (
                  <div className="error-banner" role="alert">
                    <TriangleAlert size={18} />
                    {chatError}
                  </div>
                )}
              </section>
            </>
          )}
          {tab === "tasks" && (
            <section className="panel">
              <div className="section-heading">
                <h2>От договорённостей к действиям</h2>
              </div>
              <p className="section-description">
                Исполнители, сроки и приоритеты из протокола. Неназванные сроки
                не дополняются.
              </p>
              <TasksTable items={p.action_items} />
            </section>
          )}
          {tab === "transcript" && (
            <>
              <label className="search-field">
                <Search size={17} />
                <input
                  aria-label="Поиск в расшифровке"
                  placeholder="Найти в тексте встречи…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </label>
              {audio && (
                <div className="audio-player">
                  <AudioLines size={22} />
                  <audio
                    src={audio}
                    ref={audioRef}
                    controls
                    preload="metadata"
                    onError={() => {
                      setAudio(undefined);
                      notify(
                        "Запись больше не доступна на сервере. Расшифровка сохранена.",
                      );
                    }}
                  />
                </div>
              )}
              {blocks.length ? (
                <div className="segments">
                  {blocks
                    .filter((s) =>
                      s.text.toLowerCase().includes(query.toLowerCase()),
                    )
                    .map((s, i) => (
                      <article className="segment" key={i}>
                        <span
                          className={`avatar ${["violet", "blue", "rose"][Math.max(0, speakers.indexOf(s.speaker || "")) % 3]}`}
                        >
                          {s.speaker ? (
                            s.speaker.replace("Speaker ", "S")
                          ) : (
                            <AudioLines size={15} />
                          )}
                        </span>
                        <div>
                          <div className="segment-heading">
                            <strong>
                              {s.speaker?.replace("Speaker ", "Спикер ") ||
                                "Реплика"}
                            </strong>
                            <button
                              disabled={!audio}
                              title={
                                audio
                                  ? "Воспроизвести с этого момента"
                                  : "Время в записи"
                              }
                              onClick={() => {
                                if (audioRef.current) {
                                  audioRef.current.currentTime = s.start;
                                  audioRef.current
                                    .play()
                                    .catch(() =>
                                      notify(
                                        "Нажмите воспроизведение в плеере.",
                                      ),
                                    );
                                }
                              }}
                            >
                              {duration(s.start)}
                            </button>
                            {s.language && (
                              <span className="language-tag">
                                {s.language.toUpperCase()}
                              </span>
                            )}
                          </div>
                          <p>{s.text}</p>
                        </div>
                      </article>
                    ))}
                  {!blocks.some((s) =>
                    s.text.toLowerCase().includes(query.toLowerCase()),
                  ) && (
                    <div className="empty">
                      По вашему запросу ничего не найдено.
                    </div>
                  )}
                </div>
              ) : (
                <div className="panel plain-transcript">
                  {r.transcript
                    .split(/\n+/)
                    .filter((s) =>
                      s.toLowerCase().includes(query.toLowerCase()),
                    )
                    .map((s, i) => (
                      <p key={i}>{s}</p>
                    ))}
                  {!r.transcript && <p>Сервер не вернул текст расшифровки.</p>}
                </div>
              )}
            </>
          )}
        </div>
        <aside className="result-sidebar">
          <section className="panel">
            <div className="section-heading">
              <h2>
                <Clock3 size={18} />
                Обработка встречи
              </h2>
            </div>
            <div className="timing-total">
              <strong>{secs(r.timing.total_sec)}</strong>
              <span>
                {r.filename === "(live)"
                  ? "Финализация после записи"
                  : "Полное время обработки"}
              </span>
            </div>
            <dl className="timing-list">
              <div>
                <dt>Распознавание</dt>
                <dd>{secs(r.timing.transcribe_sec)}</dd>
              </div>
              <div>
                <dt>Протокол</dt>
                <dd>{secs(r.timing.llm_sec)}</dd>
              </div>
              {r.timing.diarize_sec != null && (
                <div>
                  <dt>Диаризация</dt>
                  <dd>{secs(r.timing.diarize_sec)}</dd>
                </div>
              )}
              {r.timing.mean_latency_sec != null && (
                <div>
                  <dt>Задержка реплик</dt>
                  <dd>{secs(r.timing.mean_latency_sec)}</dd>
                </div>
              )}
            </dl>
            {r.filename !== "(live)" && r.timing.realtime_factor != null && (
              <div className="timing-factor">
                {r.timing.realtime_factor.toFixed(1)}×{" "}
                <span>длина записи / время обработки</span>
              </div>
            )}
          </section>
          {speakers.length > 0 && (
            <section className="panel">
              <div className="section-heading">
                <h2>
                  <Users size={18} />
                  Спикеры
                </h2>
              </div>
              <div className="speaker-list">
                {speakers.map((s, i) => (
                  <div key={s}>
                    <span
                      className={`avatar ${["violet", "blue", "rose"][i % 3]}`}
                    >
                      {i + 1}
                    </span>
                    <span>{s.replace("Speaker ", "Спикер ")}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
          {!!r.participants?.length && (
            <section className="panel">
              <h2>Участники встречи</h2>
              <p className="participants-text">{r.participants.join(", ")}</p>
            </section>
          )}
          <section className="result-note">
            <ShieldCheck size={23} />
            <strong>Данные остаются у вас</strong>
            <p>
              Распознавание и анализ выполняет локальный сервер. Сверяйте важные
              договорённости с расшифровкой.
            </p>
          </section>
        </aside>
      </div>
    </>
  );
}
