import { useState } from "react";
import { api, setToken } from "../api";
import { Button, Card, Field, inputClass } from "../components/ui";

/* ============================== LOGIN ============================== */
export function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await api("/v1/auth/login", "POST", { username, password });
      setToken(result.token);
      onLogin(result.user);
    } catch (err) {
      setError(err.message);
    }
    setBusy(false);
  }

  // Read-only access for evaluators, no password needed
  async function continueAsReviewer() {
    setBusy(true);
    setError("");
    try {
      const result = await api("/v1/auth/guest", "POST");
      setToken(result.token);
      onLogin(result.user);
    } catch (err) {
      setError(err.message);
    }
    setBusy(false);
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-ink-50 p-4">
      <Card className="fade-in w-full max-w-sm p-8">
        <div className="mb-6 flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-brand-600 font-display text-base font-extrabold text-white">H</div>
          <div className="leading-tight">
            <p className="font-display text-base font-bold text-ink-900">HDFC AI Platform</p>
            <p className="text-xs font-medium text-ink-400">Enterprise AI Management</p>
          </div>
        </div>
        <form onSubmit={submit} className="space-y-4">
          <Field label="Username"><input className={inputClass} value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoFocus /></Field>
          <Field label="Password"><input className={inputClass} type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" /></Field>
          {error && <p className="text-sm text-rose-600">{error}</p>}
          <Button type="submit" variant="primary" className="w-full" icon="lock" disabled={busy || !username || !password}>{busy ? "Signing in…" : "Sign in"}</Button>
        </form>
        <div className="my-5 flex items-center gap-3 text-xs text-ink-400"><span className="h-px flex-1 bg-ink-100"></span>or<span className="h-px flex-1 bg-ink-100"></span></div>
        <Button variant="secondary" className="w-full" icon="shield" onClick={continueAsReviewer} disabled={busy}>Continue as reviewer (read-only)</Button>
        <p className="mt-2 text-xs text-ink-400">For evaluators: view every page and ask the assistants, without changing anything.</p>
        <p className="mt-6 text-xs text-ink-400">Internal tool. Access depends on your role: Admin, AI Platform Engineer, Employee, or Reviewer.</p>
      </Card>
    </div>
  );
}
