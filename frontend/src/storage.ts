import { parseResult, type RecordItem } from "./api";
function db() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const r = indexedDB.open("steppe-real-meetings", 1);
    r.onupgradeneeded = () => {
      r.result.createObjectStore("records", { keyPath: "result.id" });
      r.result.createObjectStore("audio");
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function run<T>(
  store: string,
  mode: IDBTransactionMode,
  fn: (s: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction(store, mode);
    const r = fn(t.objectStore(store));
    t.oncomplete = () => {
      d.close();
      resolve(r.result);
    };
    t.onerror = () => {
      d.close();
      reject(t.error);
    };
    t.onabort = () => {
      d.close();
      reject(t.error);
    };
  });
}
export async function readRecords() {
  const records = await run<RecordItem[]>("records", "readonly", (s) =>
    s.getAll(),
  );
  return records.filter((r) => {
    try {
      parseResult(r.result);
      return true;
    } catch {
      return false;
    }
  });
}
export const saveRecord = (r: RecordItem) =>
  run("records", "readwrite", (s) => s.put(r));
export const saveAudio = (id: string, file: Blob) =>
  run("audio", "readwrite", (s) => s.put(file, id));
export const readAudio = (id: string) =>
  run<Blob | undefined>("audio", "readonly", (s) => s.get(id));
