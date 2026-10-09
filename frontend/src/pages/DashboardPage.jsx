import { useApi } from "../api";
import { Icon } from "../components/Icon";
import { Badge, Button, Card, Loading, PageHeader, SectionTitle, Stat } from "../components/ui";
import { AuditTable } from "./AuditLogsPage";

/* ============================== DASHBOARD ============================== */
export function DashboardPage({ navigate }) {
  const datasets = useApi("/v1/datasets");
  const runs = useApi("/v1/runs");
  const models = useApi("/v1/models");
  const monitoring = useApi("/v1/monitoring");
  const audit = useApi("/v1/audit");
  const health = useApi("/v1/health");
  const assistants = useApi("/v1/assistants");
  if (!datasets.data || !runs.data || !models.data || !monitoring.data || !audit.data || !assistants.data) {
    return <Loading error={datasets.error || runs.error || models.error || monitoring.error || audit.error || assistants.error} />;
  }
  const liveAssistants = assistants.data.filter((a) => a.status === "approved").length;
  const pendingAssistants = assistants.data.filter((a) => a.status === "pending").length;

  const versions = Object.values(models.data.versions);
  const approvedData = datasets.data.filter((d) => d.status === "approved").length;
  const completedRuns = runs.data.filter((r) => r.status === "completed").length;
  const queuedRuns = runs.data.filter((r) => r.status === "queued").length;
  const pendingModels = versions.filter((v) => v.status === "pending").length;
  const preparedData = datasets.data.filter((d) => d.status === "prepared").length;

  const lifecycle = [
    { title: "Data", desc: "Register & prepare data", icon: "database", page: "datasets", status: `${approvedData} approved`, ok: true },
    { title: "Training", desc: "LoRA fine-tuning", icon: "flask", page: "training", status: queuedRuns ? `${queuedRuns} queued` : `${completedRuns} completed`, ok: !queuedRuns },
    { title: "Evaluation", desc: "Quality & safety", icon: "clipboard", page: "evaluations", status: `${versions.length} evaluated`, ok: true },
    { title: "Approval", desc: "Human sign-off", icon: "shield", page: "approvals", status: `${pendingModels + preparedData + pendingAssistants} pending`, ok: pendingModels + preparedData + pendingAssistants === 0 },
    { title: "Deployment", desc: "Serve approved model", icon: "rocket", page: "deployments", status: `${models.data.live} live`, ok: true },
    { title: "Monitoring", desc: "Requests & feedback", icon: "activity", page: "monitoring", status: `${monitoring.data.requests} requests`, ok: true },
  ];

  return (
    <div className="fade-in">
      <PageHeader title="AI Platform Overview" description="The governed lifecycle from approved banking data to a served, fine-tuned model." />
      <div className="grid grid-cols-1 gap-4 min-[420px]:grid-cols-2 md:grid-cols-3 2xl:grid-cols-6">
        <Stat icon="rocket" label="Live model" value={models.data.live} sub={`Rollback to ${models.data.previous}`} />
        <Stat icon="chat" label="Assistants" value={liveAssistants} sub={`approved, ${pendingAssistants} pending`} />
        <Stat icon="flask" label="Training runs" value={runs.data.length} sub={`${completedRuns} completed, ${queuedRuns} queued`} />
        <Stat icon="shield" label="Pending approvals" value={pendingModels + preparedData + pendingAssistants} sub="Datasets, models, assistants" />
        <Stat icon="database" label="Datasets" value={datasets.data.length} sub={`${approvedData} approved`} />
        <Stat icon="activity" label="Gateway" value={health.data ? "Healthy" : "…"} sub={`${monitoring.data.requests} requests served`} />
      </div>

      <Card className="mt-6 p-6">
        <SectionTitle>AI Development Lifecycle</SectionTitle>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-6">
          {lifecycle.map((stage, i) => (
            <button key={stage.title} onClick={() => navigate(stage.page)}
              className="group flex flex-col gap-3 rounded-xl border border-ink-100 bg-white p-4 text-left shadow-card transition-all hover:-translate-y-0.5 hover:border-brand-200 hover:shadow-hover">
              <div className="flex items-center justify-between">
                <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-ink-50 text-ink-500 group-hover:bg-brand-50 group-hover:text-brand-600"><Icon name={stage.icon} size={17} /></span>
                <span className="font-display text-xs font-bold text-ink-300">{String(i + 1).padStart(2, "0")}</span>
              </div>
              <div>
                <p className="font-display text-sm font-bold text-ink-900">{stage.title}</p>
                <p className="mt-0.5 text-xs text-ink-500">{stage.desc}</p>
              </div>
              <Badge status={stage.ok ? "success" : "pending"}>{stage.status}</Badge>
            </button>
          ))}
        </div>
      </Card>

      <Card className="mt-6">
        <div className="flex items-center justify-between gap-3 border-b border-ink-100 px-4 py-4 sm:px-6">
          <h2 className="font-display text-base font-bold text-ink-900">Recent Platform Activity</h2>
          <Button variant="ghost" size="sm" onClick={() => navigate("audit")}>View all <Icon name="arrowRight" size={14} /></Button>
        </div>
        <AuditTable rows={audit.data.slice(0, 6)} />
      </Card>
    </div>
  );
}
