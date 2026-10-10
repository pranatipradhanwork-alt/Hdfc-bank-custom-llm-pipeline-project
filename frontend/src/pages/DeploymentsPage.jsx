import { useState } from "react";
import { api, formatTime, useApi } from "../api";
import { Badge, Button, Card, Loading, PageHeader, Stages, Table } from "../components/ui";

/* ============================== DEPLOYMENTS ============================== */
export function DeploymentsPage({ user, setToast }) {
  const models = useApi("/v1/models");
  const health = useApi("/v1/health");
  const monitoring = useApi("/v1/monitoring");
  const [busy, setBusy] = useState(false);
  if (!models.data || !health.data || !monitoring.data) return <Loading error={models.error || health.error || monitoring.error} />;
  const registry = models.data;
  const deployments = registry.history.filter((h) => ["promote", "rollback"].includes(h.action)).reverse();

  async function rollback() {
    if (!confirm(`Roll back production from ${registry.live} to ${registry.previous}?`)) return;
    setBusy(true);
    try {
      await api("/v1/deployments/production/rollback", "POST");
      setToast({ message: "Rolled back. The previous version is now live." });
      models.reload(); health.reload();
    } catch (err) {
      setToast({ message: err.message, type: "error" });
    }
    setBusy(false);
  }

  return (
    <div className="fade-in">
      <PageHeader title="Deployments" description="Which model version serves production, and how to go back." />
      <Card className="p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs text-ink-400">Production</p>
            <p className="font-display text-xl font-bold text-ink-900">{registry.live}</p>
            <p className="mt-1 font-mono text-xs text-ink-400">sha256 {health.data.adapter_sha256}</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge status={health.data.status === "ok" ? "healthy" : "failed"}>{health.data.status === "ok" ? "Healthy" : "Unhealthy"}</Badge>
            {user.permissions.includes("rollback") && <Button variant="danger" size="sm" icon="refresh" disabled={busy} onClick={rollback}>{busy ? "Loading model…" : `Roll back to ${registry.previous}`}</Button>}
          </div>
        </div>
        <div className="mt-5 grid grid-cols-2 gap-4 border-t border-ink-100 pt-5 text-sm sm:grid-cols-4">
          <div><p className="text-xs text-ink-400">Serving</p><p className="font-semibold text-ink-800">{health.data.model_version}</p></div>
          <div><p className="text-xs text-ink-400">Rollback target</p><p className="font-semibold text-ink-800">{registry.previous}</p></div>
          <div><p className="text-xs text-ink-400">Requests</p><p className="font-semibold tabular-nums text-ink-800">{monitoring.data.requests}</p></div>
          <div><p className="text-xs text-ink-400">Average response</p><p className="font-semibold tabular-nums text-ink-800">{monitoring.data.avg_latency_ms ? (monitoring.data.avg_latency_ms / 1000).toFixed(1) + " s" : "-"}</p></div>
        </div>
        <div className="mt-5 border-t border-ink-100 pt-5">
          <p className="mb-2 text-xs font-medium text-ink-400">Release steps</p>
          <Stages stages={["Approved", "Checksum verified", "Loaded", "Serving"]} current={4} />
        </div>
      </Card>
      <Card className="mt-6">
        <div className="border-b border-ink-100 px-6 py-4"><h2 className="font-display text-base font-bold text-ink-900">Deployment history</h2></div>
        <Table rows={deployments} emptyText="No promotions or rollbacks yet" columns={[
          { label: "Time", render: (h) => formatTime(h.time) },
          { label: "Action", render: (h) => <Badge status={h.action === "promote" ? "success" : "pending"}>{h.action}</Badge> },
          { label: "From", render: (h) => h.from || "-" },
          { label: "To", render: (h) => <span className="font-medium text-ink-900">{h.to}</span> },
          { label: "By", render: (h) => h.by || "-" },
        ]} />
      </Card>
    </div>
  );
}
