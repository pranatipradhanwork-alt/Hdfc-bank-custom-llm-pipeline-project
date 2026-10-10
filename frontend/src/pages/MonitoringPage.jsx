import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { formatTime, useApi } from "../api";
import { Badge, Card, Empty, Loading, PageHeader, RefreshButton, SectionTitle, Stat, Table } from "../components/ui";

/* ============================== MONITORING ============================== */
export function MonitoringPage() {
  const monitoring = useApi("/v1/monitoring");
  if (!monitoring.data) return <Loading error={monitoring.error} />;
  const m = monitoring.data;
  const latency = [...m.recent].reverse().map((r, i) => ({ n: i + 1, seconds: +(r.latency_ms / 1000).toFixed(1) }));

  return (
    <div className="fade-in">
      <PageHeader title="Monitoring" description="Gateway traffic, escalations, guardrail events and reviewer feedback. Customer questions are not stored, only identifiers and outcomes."
        action={<RefreshButton source={monitoring} />} />
      <Card className="mb-6 p-6">
        <SectionTitle>Service level objectives (last {m.slo_window_hours} hours)</SectionTitle>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {m.slos.map((slo) => (
            <div key={slo.name} className="rounded-lg border border-ink-100 p-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-semibold text-ink-800">{slo.name}</p>
                <Badge status={slo.status === "met" ? "passed" : slo.status === "breached" ? "failed" : "ink"}>
                  {slo.status === "met" ? "Met" : slo.status === "breached" ? "Breached" : "No data"}
                </Badge>
              </div>
              <p className="mt-3 font-display text-2xl font-bold tabular-nums text-ink-900">{slo.actual || "-"}</p>
              <p className="text-xs text-ink-400">Target {slo.target}</p>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-ink-400">The same numbers are exported for Prometheus at <span className="font-mono">/metrics</span>.</p>
      </Card>
      <div className="grid grid-cols-1 gap-4 min-[420px]:grid-cols-2 xl:grid-cols-4">
        <Stat icon="activity" label="Requests" value={m.requests} sub={`${m.answered} answered, ${m.escalated} escalated, ${m.failed} failed`} />
        <Stat icon="refresh" label="Answer time p50 / p95" value={m.p50_latency_ms ? `${(m.p50_latency_ms / 1000).toFixed(1)} / ${(m.p95_latency_ms / 1000).toFixed(1)} s` : "-"} sub={m.avg_latency_ms ? `average ${(m.avg_latency_ms / 1000).toFixed(1)} s` : ""} />
        <Stat icon="shield" label="Guardrail events" value={Object.values(m.policy_flags).reduce((a, b) => a + b, 0)} sub={Object.keys(m.policy_flags).length ? Object.keys(m.policy_flags).join(", ").replace(/_/g, " ") : "none"} />
        <Stat icon="thumbUp" label="Feedback" value={`${m.feedback_good} / ${m.feedback_bad}`} sub="good / bad" />
      </div>
      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="p-6">
          <SectionTitle>Response time, last {latency.length} requests (s)</SectionTitle>
          {latency.length ? (
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={latency} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke="#EEEEF1" />
                  <XAxis dataKey="n" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} width={34} />
                  <Tooltip contentStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="seconds" stroke="#0F9B8E" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <Empty icon="activity" title="No requests yet" sub="Ask the assistant a question to see traffic here." />}
        </Card>
        <Card className="p-6">
          <SectionTitle>Guardrail events by type</SectionTitle>
          {Object.keys(m.policy_flags).length ? Object.entries(m.policy_flags).map(([flag, count]) => (
            <div key={flag} className="flex items-center justify-between border-b border-ink-50 py-2 text-sm last:border-0">
              <span className="text-ink-700">{flag.replace(/_/g, " ")}</span><span className="font-semibold tabular-nums text-ink-900">{count}</span>
            </div>
          )) : <Empty icon="shield" title="No guardrail events" />}
        </Card>
      </div>
      <Card className="mt-6">
        <div className="border-b border-ink-100 px-6 py-4"><h2 className="font-display text-base font-bold text-ink-900">Recent requests</h2></div>
        <Table rows={m.recent} emptyText="No requests yet" columns={[
          { label: "Time", render: (r) => formatTime(r.time) },
          { label: "Trace", render: (r) => <span className="font-mono text-xs">{r.trace_id.slice(0, 12)}</span> },
          { label: "Caller", render: (r) => r.user },
          { label: "Model", render: (r) => r.model_version },
          { label: "Confidence", render: (r) => r.confidence ? <Badge status={r.confidence} /> : "-" },
          { label: "Outcome", render: (r) => r.ok === false ? <Badge status="failed">failed</Badge> : r.escalation_required ? <Badge status="escalated">escalated</Badge> : <Badge status="success">answered</Badge> },
          { label: "Latency", render: (r) => `${(r.latency_ms / 1000).toFixed(1)} s` },
        ]} />
      </Card>
    </div>
  );
}
