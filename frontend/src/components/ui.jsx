import React from "react";
import { Icon } from "./Icon";

/* ------------------------------ Shared UI ------------------------------ */
export function tone(status) {
  const s = (status || "").toLowerCase();
  if (["approved","success","completed","live","healthy","passed","ok","good","high"].includes(s)) return "emerald";
  if (["rejected","failed","denied","killed","bad","low"].includes(s)) return "rose";
  if (["pending","preparing","prepared","registered","queued","running","medium","escalated"].includes(s)) return "amber";
  return "ink";
}
export const TONE_CLASSES = {
  emerald: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
  rose: "bg-rose-50 text-rose-700 ring-rose-600/20",
  amber: "bg-amber-50 text-amber-700 ring-amber-600/20",
  ink: "bg-ink-100 text-ink-600 ring-ink-300/40",
};
export const DOT_CLASSES = { emerald: "bg-emerald-500", rose: "bg-rose-500", amber: "bg-amber-500", ink: "bg-ink-400" };

export function Badge({ status, children }) {
  const t = tone(status);
  return (
    <span className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${children ? "" : "capitalize"} ${TONE_CLASSES[t]}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${DOT_CLASSES[t]}`} />
      {children || status}
    </span>
  );
}

export function Card({ children, className = "", hover = false, onClick }) {
  return (
    <div onClick={onClick}
      className={`rounded-xl border border-ink-100 bg-white shadow-card ${hover ? "cursor-pointer transition-all duration-150 hover:border-ink-200 hover:shadow-hover" : ""} ${className}`}>
      {children}
    </div>
  );
}

export function Button({ children, variant = "secondary", size = "md", className = "", icon, onClick, disabled, type = "button" }) {
  const sizes = { sm: "px-3 py-1.5 text-xs", md: "px-4 py-2 text-sm" };
  const variants = {
    primary: "bg-brand-600 text-white hover:bg-brand-700 shadow-sm",
    secondary: "bg-white text-ink-700 ring-1 ring-inset ring-ink-200 hover:bg-ink-50",
    danger: "bg-white text-rose-600 ring-1 ring-inset ring-rose-200 hover:bg-rose-50",
    ghost: "bg-transparent text-ink-500 hover:bg-ink-100",
    success: "bg-emerald-600 text-white hover:bg-emerald-700 shadow-sm",
  };
  return (
    <button type={type} onClick={onClick} disabled={disabled}
      className={`inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40 ${sizes[size]} ${variants[variant]} ${className}`}>
      {icon && <Icon name={icon} size={size === "sm" ? 14 : 16} />}
      {children}
    </button>
  );
}

export function PageHeader({ title, description, action }) {
  return (
    <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
      <div>
        <h1 className="font-display text-xl font-bold text-ink-900 sm:text-2xl">{title}</h1>
        {description && <p className="mt-1 text-sm text-ink-500">{description}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

export function SectionTitle({ children, action }) {
  return (
    <div className="mb-4 flex items-center justify-between">
      <h2 className="font-display text-base font-bold text-ink-900">{children}</h2>
      {action}
    </div>
  );
}

export function Breadcrumb({ items }) {
  return (
    <div className="mb-3 flex items-center gap-1.5 text-sm text-ink-500">
      {items.map((it, i) => (
        <React.Fragment key={i}>
          {i > 0 && <Icon name="chevronRight" size={14} className="text-ink-300" />}
          {it.onClick
            ? <button onClick={it.onClick} className="font-medium hover:text-brand-600">{it.label}</button>
            : <span className="font-medium text-ink-800">{it.label}</span>}
        </React.Fragment>
      ))}
    </div>
  );
}

export function Loading({ error }) {
  if (error) return <Card className="p-6 text-sm text-rose-600">Could not load: {error}</Card>;
  return <p className="py-16 text-center text-sm text-ink-400">Loading…</p>;
}

export function Empty({ icon, title, sub }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-ink-100 text-ink-400"><Icon name={icon} size={22} /></div>
      <p className="font-medium text-ink-700">{title}</p>
      {sub && <p className="text-sm text-ink-400">{sub}</p>}
    </div>
  );
}

export function Toast({ toast }) {
  if (!toast) return null;
  const isError = toast.type === "error";
  return (
    <div className="fade-in fixed bottom-4 left-4 right-4 z-50 flex max-w-md sm:bottom-6 sm:left-auto sm:right-6 items-center gap-3 rounded-xl border border-ink-100 bg-white px-4 py-3 shadow-hover">
      <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${isError ? "bg-rose-100 text-rose-600" : "bg-emerald-100 text-emerald-600"}`}>
        <Icon name={isError ? "x" : "check"} size={14} />
      </span>
      <p className="text-sm font-medium text-ink-800">{toast.message}</p>
    </div>
  );
}

export function Stages({ stages, current, failed = false }) {
  // current = index of the step in progress; steps before it are done
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-3">
      {stages.map((s, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <React.Fragment key={s}>
            <div className="flex items-center gap-1.5">
              <span className={`flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-semibold ${
                done ? "bg-emerald-500 text-white" : active && failed ? "bg-rose-500 text-white" : active ? "bg-brand-100 text-brand-700 ring-2 ring-brand-500" : "bg-ink-100 text-ink-400"}`}>
                {done ? <Icon name="check" size={12} /> : active && failed ? <Icon name="x" size={12} /> : i + 1}
              </span>
              <span className={`text-xs font-medium ${done || active ? "text-ink-800" : "text-ink-400"}`}>{s}</span>
            </div>
            {i < stages.length - 1 && <Icon name="chevronRight" size={14} className="text-ink-300" />}
          </React.Fragment>
        );
      })}
    </div>
  );
}

export function Table({ columns, rows, onRowClick, emptyText = "Nothing here yet" }) {
  // columns: [{ label, render(row) }]
  if (!rows.length) return <Empty icon="file" title={emptyText} />;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead>
          <tr className="text-xs uppercase tracking-wide text-ink-400">
            {columns.map((c, i) => <th key={i} className={`py-3 font-medium ${i === 0 ? "px-6" : "px-3"}`}>{c.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r} onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={`border-t border-ink-50 hover:bg-ink-25 ${onRowClick ? "cursor-pointer" : ""}`}>
              {columns.map((c, i) => <td key={i} className={`py-3.5 text-ink-600 ${i === 0 ? "px-6" : "px-3"}`}>{c.render(row)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Modal({ title, onClose, children }) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-ink-900/40 p-4" onClick={onClose}>
      <div className="fade-in max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl bg-white p-6 shadow-hover" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="font-display text-lg font-bold text-ink-900">{title}</h2>
          <button onClick={onClose} className="text-ink-400 hover:text-ink-700"><Icon name="x" /></button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function Field({ label, children, hint }) {
  return (
    <label className="block text-sm">
      <span className="font-medium text-ink-700">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-1 block text-xs text-ink-400">{hint}</span>}
    </label>
  );
}
export const inputClass = "w-full rounded-lg border border-ink-200 bg-white px-3 py-2 text-sm text-ink-800";

export function Info({ items }) {
  // items: [[label, value]]
  return (
    <dl className="space-y-3 text-sm">
      {items.map(([k, v]) => (
        <div key={k} className="flex items-start justify-between gap-4">
          <dt className="shrink-0 text-ink-400">{k}</dt>
          <dd className="break-all text-right font-medium text-ink-800">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Stat({ icon, label, value, sub }) {
  return (
    <Card className="p-5">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-brand-50 text-brand-600"><Icon name={icon} size={19} /></div>
      <p className="mt-4 font-display text-xl font-bold tabular-nums text-ink-900 sm:text-2xl">{value}</p>
      <p className="text-sm text-ink-500">{label}</p>
      {sub && <p className="mt-2 text-xs font-medium text-ink-400">{sub}</p>}
    </Card>
  );
}

export function RefreshButton({ source }) {
  return (
    <div className="flex items-center gap-3">
      {source.updatedAt && <span className="text-xs text-ink-400">Updated {source.updatedAt.toLocaleTimeString("en-IN")}</span>}
      <Button icon="refresh" disabled={source.loading} onClick={source.reload}>{source.loading ? "Refreshing…" : "Refresh"}</Button>
    </div>
  );
}

export function CommandHint({ command }) {
  return (
    <div className="mt-3 rounded-lg bg-ink-900 px-4 py-3 font-mono text-xs text-emerald-300">
      <p className="mb-1 flex items-center gap-1.5 font-sans text-[11px] font-medium text-ink-300"><Icon name="terminal" size={12} /> Run on the GPU worker</p>
      {command}
    </div>
  );
}
