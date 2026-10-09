import { useState, useEffect } from "react";

/* ------------------------------ API helper ------------------------------ */
export const TOKEN_KEY = "hdfc_token";

// The login token lives in sessionStorage, which belongs to one tab, so each tab can be signed in as a different
// role (e.g. admin in one tab, employee in another). localStorage would be shared by every tab of the site.
export function getToken() {
  try { return sessionStorage.getItem(TOKEN_KEY); } catch { return null; }
}
export function setToken(token) {
  try { token ? sessionStorage.setItem(TOKEN_KEY, token) : sessionStorage.removeItem(TOKEN_KEY); } catch {}
}

// Calls the API with the login token and turns error responses into readable messages
export async function api(path, method = "GET", body) {
  const headers = { "Content-Type": "application/json" };
  if (getToken()) headers.Authorization = "Bearer " + getToken();
  const response = await fetch(path, { method, headers, body: body ? JSON.stringify(body) : undefined });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && path !== "/v1/auth/login") window.dispatchEvent(new Event("session-expired"));
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join(", ") : data.detail;
    throw new Error(detail || response.statusText);
  }
  return data;
}

// Loads data for a page; call reload() after a change to fetch it again
export function useApi(path) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [version, setVersion] = useState(0);
  const [loading, setLoading] = useState(true);
  const [updatedAt, setUpdatedAt] = useState(null);
  useEffect(() => {
    let active = true;
    setError(null);
    setLoading(true);
    api(path)
      .then((d) => { if (active) { setData(d); setUpdatedAt(new Date()); } })
      .catch((e) => active && setError(e.message))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, [path, version]);
  return { data, error, loading, updatedAt, reload: () => setVersion((v) => v + 1) };
}

export function formatTime(iso) {
  if (!iso) return "-";
  return new Date(iso).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}
