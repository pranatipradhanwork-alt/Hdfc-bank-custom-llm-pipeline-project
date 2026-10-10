import { useState } from "react";
import { formatTime, useApi } from "../api";
import { Badge, Card, Loading, PageHeader, RefreshButton, Table } from "../components/ui";

/* ============================== AUDIT LOGS ============================== */
export function AuditTable({ rows }) {
  return <Table rows={rows} emptyText="No activity recorded yet" columns={[
    { label: "Time", render: (l) => <span className="tabular-nums text-ink-500">{formatTime(l.time)}</span> },
    { label: "User", render: (l) => <span className="font-medium text-ink-800">{l.user}</span> },
    { label: "Action", render: (l) => l.action.replace(/_/g, " ") },
    { label: "Resource", render: (l) => l.resource },
    { label: "Result", render: (l) => <Badge status={l.result} /> },
    { label: "Detail", render: (l) => <span className="text-xs text-ink-400">{l.detail}</span> },
  ]} />;
}

export function AuditLogsPage() {
  const audit = useApi("/v1/audit");
  const [userFilter, setUserFilter] = useState("");
  const [resultFilter, setResultFilter] = useState("");
  if (!audit.data) return <Loading error={audit.error} />;
  const users = [...new Set(audit.data.map((l) => l.user))];
  const rows = audit.data.filter((l) => (!userFilter || l.user === userFilter) && (!resultFilter || l.result === resultFilter));

  return (
    <div className="fade-in">
      <PageHeader title="Audit Logs" description="Every login and governance action: who did what, to which resource, and the result."
        action={<RefreshButton source={audit} />} />
      <div className="mb-4 flex flex-wrap items-center gap-2.5">
        <select className="rounded-lg border border-ink-200 bg-white px-3 py-2 text-sm" value={userFilter} onChange={(e) => setUserFilter(e.target.value)}>
          <option value="">All users</option>{users.map((u) => <option key={u}>{u}</option>)}
        </select>
        <select className="rounded-lg border border-ink-200 bg-white px-3 py-2 text-sm" value={resultFilter} onChange={(e) => setResultFilter(e.target.value)}>
          <option value="">All results</option>{["success", "failed", "denied"].map((r) => <option key={r}>{r}</option>)}
        </select>
        <span className="ml-auto text-xs text-ink-400">{rows.length} of {audit.data.length} entries</span>
      </div>
      <Card><AuditTable rows={rows} /></Card>
    </div>
  );
}
