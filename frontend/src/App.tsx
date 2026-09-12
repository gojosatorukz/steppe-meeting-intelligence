import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  AudioLines,
  CheckCheck,
  ChevronRight,
  Clock3,
  FileText,
  HardDrive,
  LayoutGrid,
  ListTodo,
  LoaderCircle,
  LockKeyhole,
  Menu,
  Mic,
  Plus,
  RefreshCw,
  Search,
  Server,
  Settings2,
  ShieldCheck,
  UploadCloud,
  X,
} from "lucide-react";
import * as api from "./api";
import type { Health, RecordItem, Result } from "./api";
import { readRecords, saveRecord, saveAudio } from "./storage";
import { Upload } from "./components/Upload";
import { Detail, TasksTable } from "./components/Detail";
import { Live } from "./components/Live";
type View = "dashboard" | "meetings" | "tasks" | "live" | "settings";
const labels: Record<View, string> = {
  dashboard: "Обзор",
  meetings: "Мои встречи",
  tasks: "Поручения",
  live: "Живая встреча",
  settings: "Настройки",
};
export default function App() {
  const [records, setRecords] = useState<RecordItem[]>([]),
    [view, setView] = useState<View>(
      location.pathname === "/live" ? "live" : "dashboard",
    ),
    [selected, setSelected] = useState<string | null>(null),
    [newOpen, setNewOpen] = useState(false),
    [health, setHealth] = useState<Health | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [toast, setToast] = useState(""),
    [query, setQuery] = useState(""),
    [taskQuery, setTaskQuery] = useState(""),
    [mobile, setMobile] = useState(false),
    [liveState, setLiveState] = useState("idle");
  const refreshing = useRef(false);
  const current = records.find((x) => x.result.id === selected);
  const tasks = records.flatMap((x) => x.result.protocol.action_items);
  const activeLive = liveState === "running" || liveState === "finalizing";
  const refresh = useCallback(async () => {
    if (refreshing.current) return;
    refreshing.current = true;
    setBusy(true);
    setError("");
    try {
      const [h, list] = await Promise.allSettled([api.health(), api.results()]);
      setHealth(h.status === "fulfilled" ? h.value : null);
      if (list.status === "rejected") throw list.reason;
      const fresh: Result[] = [];
      let failed = 0;
      for (let i = 0; i < list.value.length; i += 4) {
        const batch = await Promise.allSettled(
          list.value.slice(i, i + 4).map((row) => api.result(row.id)),
        );
        batch.forEach((x) => {
          if (x.status === "fulfilled") fresh.push(x.value);
          else failed++;
        });
      }
      setRecords((prev) => {
        const map = new Map(prev.map((x) => [x.result.id, x]));
        fresh.forEach((r) => {
          const old = map.get(r.id);
          const entry = {
            result: r,
            title: old?.title || api.titleOf(r),
            receivedAt: old?.receivedAt,
          };
          map.set(r.id, entry);
          saveRecord(entry).catch(() =>
            setToast(
              "Не удалось сохранить локальную копию. Скачайте важные результаты.",
            ),
          );
        });
        return [...map.values()].sort((a, b) =>
          (b.receivedAt || "").localeCompare(a.receivedAt || ""),
        );
      });
      if (failed)
        setError(
          `Не удалось загрузить ${failed} результатов. Обновите список ещё раз.`,
        );
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Не удалось загрузить встречи.",
      );
    } finally {
      setBusy(false);
      refreshing.current = false;
    }
  }, []);
  useEffect(() => {
    let dead = false;
    readRecords()
      .then((rows) => {
        if (!dead) setRecords(rows);
      })
      .catch(() =>
        setToast("Хранилище недоступно. Экспортируйте важные результаты."),
      )
      .finally(() => {
        if (!dead) refresh();
      });
    return () => {
      dead = true;
    };
  }, [refresh]);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 8000);
    return () => clearTimeout(t);
  }, [toast]);
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMobile(false);
    };
    document.addEventListener("keydown", f);
    return () => document.removeEventListener("keydown", f);
  }, []);
  function navigate(v: View) {
    setView(v);
    setSelected(null);
    setMobile(false);
    setQuery("");
    window.scrollTo({ top: 0 });
  }
  function open(r: RecordItem) {
    setSelected(r.result.id);
    setView("meetings");
    setMobile(false);
    window.scrollTo({ top: 0 });
  }
  async function complete(r: Result, title: string, file?: File) {
    const entry: RecordItem = {
      result: r,
      title: title || api.titleOf(r),
      receivedAt: new Date().toISOString(),
    };
    setRecords((prev) => [entry, ...prev.filter((x) => x.result.id !== r.id)]);
    try {
      await saveRecord(entry);
      if (file) await saveAudio(r.id, file);
    } catch {
      setToast(
        "Встреча обработана, но копию сохранить не удалось. Скачайте JSON.",
      );
    }
    setNewOpen(false);
    open(entry);
  }
  const filtered = records.filter((r) =>
    [r.title, r.result.filename || ""]
      .join(" ")
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const visibleTasks = records
    .map((r) => ({
      ...r,
      items: r.result.protocol.action_items.filter((t) =>
        [t.task, t.owner, t.due]
          .join(" ")
          .toLowerCase()
          .includes(taskQuery.toLowerCase()),
      ),
    }))
    .filter((r) => r.items.length);
  return (
    <div className="app-shell">
      {mobile && (
        <button
          className="mobile-backdrop"
          aria-label="Закрыть меню"
          onClick={() => setMobile(false)}
        />
      )}
      <aside className={`sidebar ${mobile ? "mobile-open" : ""}`}>
        <button
          className="brand"
          aria-label="Steppe — обзор"
          onClick={() => navigate("dashboard")}
        >
          <span>
            <AudioLines size={25} />
          </span>
          steppe<span className="brand-period">.</span>
        </button>
        <div className="workspace-switch">
          <span className="workspace-icon">Г</span>
          <div>
            <strong>Команда Гонконг</strong>
            <span>Локальное пространство</span>
          </div>
        </div>
        <div className="nav-caption">ПРОСТРАНСТВО</div>
        <nav aria-label="Основная навигация">
          {(
            [
              ["dashboard", LayoutGrid],
              ["meetings", FileText],
              ["tasks", ListTodo],
              ["live", Mic],
            ] as const
          ).map(([v, Icon]) => (
            <button
              className={`nav-item ${view === v ? "active" : ""}`}
              key={v}
              onClick={() => navigate(v)}
              aria-current={view === v ? "page" : undefined}
            >
              <Icon size={19} />
              {labels[v]}
              {v === "meetings" && records.length > 0 && (
                <span>{records.length}</span>
              )}
              {v === "tasks" && tasks.length > 0 && <span>{tasks.length}</span>}
              {v === "live" && activeLive && <i className="record-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy-box">
            <span className="privacy-icon">
              <ShieldCheck size={22} />
            </span>
            <strong>Ваши встречи — ваши данные</strong>
            <p>
              Распознавание и анализ
              <br />
              на вашем устройстве.
            </p>
            <span>
              <LockKeyhole size={12} />
              100% LOCAL
            </span>
          </div>
          <button
            className={`nav-item ${view === "settings" ? "active" : ""}`}
            onClick={() => navigate("settings")}
          >
            <Settings2 size={18} />
            Настройки
          </button>
          <div className="profile">
            <span className="avatar violet">Г</span>
            <div>
              <strong>AI Steppe Tech Hack</strong>
              <span>Meeting Intelligence</span>
            </div>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <span className="breadcrumb">
            <button
              className="mobile-menu icon-button"
              aria-label="Открыть меню"
              aria-expanded={mobile}
              onClick={() => setMobile(true)}
            >
              <Menu size={20} />
            </button>
            <span className="breadcrumb-root">
              Пространство <span className="slash">/</span>
            </span>
            <strong>{current ? "Протокол встречи" : labels[view]}</strong>
          </span>
          <div>
            {activeLive && view !== "live" && (
              <button
                className="pill recording-pill"
                onClick={() => navigate("live")}
              >
                <i className="record-dot" />
                Идёт запись
              </button>
            )}
            <span className={`pill ${health?.ok ? "local" : "neutral"}`}>
              <Server size={13} />
              {health?.ok
                ? "Сервер подключён"
                : busy
                  ? "Подключение…"
                  : "Сервер недоступен"}
            </span>
          </div>
        </header>
        <main>
          <Live
            active={view === "live" && !current}
            onComplete={complete}
            onState={setLiveState}
          />
          {current ? (
            <Detail
              record={current}
              onBack={() => navigate("meetings")}
              notify={setToast}
            />
          ) : view === "live" ? null : view === "settings" ? (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">СЕРВЕР И МОДЕЛИ</div>
                  <h1>Локальное пространство</h1>
                  <p>Состояние обработки на вашем компьютере.</p>
                </div>
                <button
                  className="button secondary"
                  onClick={refresh}
                  disabled={busy}
                >
                  <RefreshCw size={16} className={busy ? "spin" : ""} />
                  Проверить
                </button>
              </div>
              <div className="settings-grid">
                <section className="panel">
                  <div className="section-heading">
                    <h2>
                      <Server size={19} />
                      Подключение
                    </h2>
                    <span
                      className={`pill ${health?.ok ? "local" : "neutral"}`}
                    >
                      {health?.ok ? "Подключено" : "Нет связи"}
                    </span>
                  </div>
                  <dl className="settings-list">
                    {[
                      ["Распознавание речи", api.sttLabel(health)],
                      ["Языковая модель", health?.llm_model || "Нет данных"],
                      [
                        "Диаризация",
                        health?.diarization
                          ? health.diarization.available
                            ? "Доступна"
                            : "Модели не найдены"
                          : "Не проверено",
                      ],
                      [
                        "Ollama",
                        health
                          ? health.ollama_running
                            ? "Запущена"
                            : "Не запущена"
                          : "Не проверено",
                      ],
                      [
                        "Готовность LLM",
                        health
                          ? health.llm_ready
                            ? "Готова"
                            : "Модель не найдена"
                          : "Не проверено",
                      ],
                      [
                        "Результатов в памяти сервера",
                        health ? String(health.results_stored) : "—",
                      ],
                    ].map(([k, v]) => (
                      <div key={k}>
                        <dt>{k}</dt>
                        <dd>{v}</dd>
                      </div>
                    ))}
                  </dl>
                  {!!health?.ollama_models?.length && (
                    <p className="field-hint">
                      Установлены: {health.ollama_models.join(", ")}
                    </p>
                  )}
                  {error && (
                    <div className="error-banner" role="alert">
                      {error}
                    </div>
                  )}
                </section>
                <section className="panel privacy-settings">
                  <span className="stat-icon lavender">
                    <HardDrive size={23} />
                  </span>
                  <h2>Всё остаётся на устройстве</h2>
                  <p>
                    Протоколы сохраняются в браузере. Сервер хранит результаты
                    до перезапуска. Загруженные аудиофайлы доступны в локальном
                    плеере.
                  </p>
                  <div className="settings-note">
                    <ShieldCheck size={18} />
                    <span>
                      Для автономной работы модели должны быть заранее
                      установлены. Статус выше приходит от настоящего backend.
                    </span>
                  </div>
                  <div className="settings-note">
                    Не очищайте данные браузера до экспорта важных встреч.
                  </div>
                </section>
              </div>
            </>
          ) : view === "tasks" ? (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">ДОГОВОРЁННОСТИ В ДЕЙСТВИИ</div>
                  <h1>Поручения</h1>
                  <p>
                    {tasks.length
                      ? `${tasks.length} поручений из ваших встреч.`
                      : "Поручения появятся после обработки вашей встречи."}
                  </p>
                </div>
              </div>
              <label className="search-field tasks-search">
                <Search size={17} />
                <input
                  value={taskQuery}
                  onChange={(e) => setTaskQuery(e.target.value)}
                  placeholder="Задача, ответственный или срок…"
                  aria-label="Поиск поручений"
                />
              </label>
              {visibleTasks.map((r) => (
                <section className="panel task-group" key={r.result.id}>
                  <div className="section-heading">
                    <button className="group-title" onClick={() => open(r)}>
                      <FileText size={17} />
                      {r.title}
                      <ChevronRight size={16} />
                    </button>
                    <span className="count">{r.items.length}</span>
                  </div>
                  <TasksTable items={r.items} />
                </section>
              ))}
              {!visibleTasks.length && (
                <div className="empty panel">
                  <ListTodo size={32} />
                  <h3>
                    {taskQuery ? "Поручения не найдены" : "Пока нет поручений"}
                  </h3>
                  <p>
                    {taskQuery
                      ? "Измените запрос."
                      : "Загрузите аудио или начните живую встречу."}
                  </p>
                  {!taskQuery && (
                    <button
                      className="button primary"
                      onClick={() => setNewOpen(true)}
                    >
                      Добавить встречу
                    </button>
                  )}
                </div>
              )}
            </>
          ) : (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">ВАШЕ РАБОЧЕЕ ПРОСТРАНСТВО</div>
                  <h1>
                    {view === "meetings"
                      ? "Мои встречи"
                      : "Встречи, из которых всё ясно."}
                  </h1>
                  <p>
                    {view === "meetings"
                      ? "Ваши записи, решения и планы действий."
                      : "Реальные разговоры. Понятные решения. Следующие шаги."}
                  </p>
                </div>
                <button
                  className="button primary"
                  onClick={() => setNewOpen(true)}
                >
                  <Plus size={18} />
                  Новая встреча
                </button>
              </div>
              {error && (
                <div className="connection-banner" role="alert">
                  <AlertCircle size={18} />
                  <span>{error}</span>
                  <button onClick={refresh} disabled={busy}>
                    Повторить
                  </button>
                </div>
              )}
              {health && !health.llm_ready && (
                <div className="connection-banner warning">
                  <AlertCircle size={18} />
                  <span>
                    {health.ollama_running
                      ? "Сервер работает, но выбранная LLM не найдена в Ollama."
                      : "Сервер работает. Запустите Ollama, чтобы формировать протоколы."}
                  </span>
                  <button onClick={() => navigate("settings")}>
                    Подробнее
                  </button>
                </div>
              )}
              {view === "dashboard" && (
                <>
                  {records.length > 0 && (
                    <section className="stats">
                      <div>
                        <span className="stat-icon lavender">
                          <AudioLines size={21} />
                        </span>
                        <div>
                          <span>Встреч в пространстве</span>
                          <strong>{records.length}</strong>
                        </div>
                      </div>
                      <div>
                        <span className="stat-icon mint">
                          <CheckCheck size={21} />
                        </span>
                        <div>
                          <span>Принято решений</span>
                          <strong>
                            {records.reduce(
                              (n, r) => n + r.result.protocol.decisions.length,
                              0,
                            )}
                          </strong>
                        </div>
                      </div>
                      <div>
                        <span className="stat-icon peach">
                          <ListTodo size={21} />
                        </span>
                        <div>
                          <span>Поручений</span>
                          <strong>{tasks.length}</strong>
                        </div>
                      </div>
                    </section>
                  )}
                  <section className="capture-card real-capture">
                    <div className="capture-copy">
                      <span className="pill light">
                        <AudioLines size={14} />
                        AI MEETING INTELLIGENCE
                      </span>
                      <h2>
                        Вы обсуждаете.
                        <br />
                        Steppe фиксирует главное.
                      </h2>
                      <p>
                        Добавьте аудиозапись или текст —<br />
                        получите протокол вашей встречи.
                      </p>
                      <div className="capture-foot">
                        <LockKeyhole size={14} />
                        Без облака и внешних API
                      </div>
                    </div>
                    <button
                      className="drop-preview"
                      onClick={() => setNewOpen(true)}
                    >
                      <span className="upload-symbol">
                        <UploadCloud size={28} />
                      </span>
                      <strong>Добавить запись встречи</strong>
                      <span>Аудиофайл или готовый текст</span>
                      <div className="file-types">
                        <i>MP3</i>
                        <i>WAV</i>
                        <i>M4A</i>
                      </div>
                      <span className="button white">
                        Выбрать файл <Plus size={16} />
                      </span>
                    </button>
                  </section>
                  <button
                    className="live-entry"
                    onClick={() => navigate("live")}
                  >
                    <span className="stat-icon lavender">
                      <Mic size={22} />
                    </span>
                    <span>
                      <strong>
                        {activeLive
                          ? "Вернуться к текущей записи"
                          : "Встреча начинается сейчас?"}
                      </strong>
                      <span>
                        {activeLive
                          ? "Показать живую расшифровку"
                          : "Записывайте с микрофона и следите за расшифровкой в реальном времени."}
                      </span>
                    </span>
                    <ArrowRight size={20} />
                  </button>
                </>
              )}
              <section className="meeting-section">
                <div className="section-heading">
                  <h2>
                    {view === "meetings" ? "Все встречи" : "Последние встречи"}
                    {records.length > 0 && (
                      <span className="count">{records.length}</span>
                    )}
                  </h2>
                  <button
                    className="refresh-button"
                    onClick={refresh}
                    disabled={busy}
                  >
                    <RefreshCw size={15} className={busy ? "spin" : ""} />
                    {busy ? "Обновляем…" : "Обновить"}
                  </button>
                </div>
                {records.length > 0 && (
                  <label className="search-field">
                    <Search size={17} />
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Найти встречу…"
                      aria-label="Поиск встреч"
                    />
                  </label>
                )}
                <div className="meeting-list real-meeting-list">
                  {filtered.map((record) => (
                    <button
                      key={record.result.id}
                      className="meeting-row"
                      onClick={() => open(record)}
                    >
                      <span className="meeting-icon">
                        {record.result.filename === "(live)" ? (
                          <Mic size={24} />
                        ) : (
                          <AudioLines size={24} />
                        )}
                      </span>
                      <span className="meeting-row-content">
                        <strong>{record.title}</strong>
                        <span className="row-meta">
                          <span>
                            <Clock3 size={13} />
                            {record.result.audio_seconds
                              ? api.duration(record.result.audio_seconds)
                              : "Текст"}
                          </span>
                          <span>
                            {api.languageName(record.result.language)}
                          </span>
                          <span>
                            {record.result.protocol.decisions.length} решений
                          </span>
                          <span>
                            {record.result.protocol.action_items.length}{" "}
                            поручений
                          </span>
                        </span>
                        {record.result.protocol.summary[0] && (
                          <span className="meeting-preview">
                            {record.result.protocol.summary[0]}
                          </span>
                        )}
                      </span>
                      <ChevronRight size={19} />
                    </button>
                  ))}
                </div>
                {!filtered.length && (
                  <div className="empty actual-empty">
                    {busy ? (
                      <LoaderCircle className="spin" size={26} />
                    ) : (
                      <FileText size={28} />
                    )}
                    <h3>
                      {busy
                        ? "Загружаем ваши встречи"
                        : query
                          ? "Ничего не найдено"
                          : "Ваша первая встреча — здесь"}
                    </h3>
                    <p>
                      {query
                        ? "Попробуйте другое название."
                        : "Добавьте запись, вставьте текст или включите микрофон. Здесь появятся только ваши результаты."}
                    </p>
                  </div>
                )}
              </section>
              <footer className="workspace-footer">
                <span>
                  <ShieldCheck size={14} />
                  На вашем устройстве. Под вашим контролем.
                </span>
                <span>STEPPE / MEETING INTELLIGENCE</span>
              </footer>
            </>
          )}
        </main>
      </div>
      {newOpen && (
        <Upload onClose={() => setNewOpen(false)} onComplete={complete} />
      )}
      <div
        className={`toast ${toast ? "visible" : ""}`}
        role="status"
        aria-live="polite"
      >
        {toast && (
          <>
            <span>{toast}</span>
            <button
              aria-label="Закрыть уведомление"
              onClick={() => setToast("")}
            >
              <X size={16} />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
