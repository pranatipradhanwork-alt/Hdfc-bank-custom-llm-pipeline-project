import { useApi } from "../api";
import { Badge, Card, Empty, Loading, PageHeader } from "../components/ui";

/* ============================== APPROVALS ============================== */
export function ApprovalsPage({ navigate, user }) {
  const models = useApi("/v1/models");
  const datasets = useApi("/v1/datasets");
  const assistants = useApi("/v1/assistants");
  if (!models.data || !datasets.data || !assistants.data) return <Loading error={models.error || datasets.error || assistants.error} />;
  const pendingModels = Object.entries(models.data.versions).filter(([, m]) => m.status === "pending");
  const preparedData = datasets.data.filter((d) => d.status === "prepared");
  const pendingAssistants = assistants.data.filter((a) => a.status === "pending");
  const canApprove = user.permissions.includes("review_model");

  return (
    <div className="fade-in">
      <PageHeader title="Approval Center" description="Human sign-off before data is used for training or a model goes live."
        action={!canApprove && <Badge status="pending">View only: approvals need the Admin role</Badge>} />
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {pendingModels.map(([name, m]) => (
          <Card key={name} hover onClick={() => navigate("modelDetail", name)} className="p-5">
            <div className="flex items-start justify-between"><div><p className="text-xs text-ink-400">Model</p><p className="font-display text-base font-bold text-ink-900">{name}</p></div><Badge status="pending" /></div>
            <p className="mt-3 text-sm text-ink-500">From run {m.run_id}. ROUGE-L (reworded) {m.evaluation.rougeL_finetuned_with_rag_reworded ?? "-"}.</p>
          </Card>
        ))}
        {preparedData.map((d) => (
          <Card key={d.id} hover onClick={() => navigate("datasetDetail", d.id)} className="p-5">
            <div className="flex items-start justify-between"><div><p className="text-xs text-ink-400">Dataset</p><p className="font-display text-base font-bold text-ink-900">{d.name}</p></div><Badge status="prepared" /></div>
            <p className="mt-3 text-sm text-ink-500">Delta v{d.delta_version}, {d.quality_report.final_rows} rows{d.parent_id ? `, selected from ${d.parent_id}` : `, leakage ${d.quality_report.contamination["test_rows_at_or_above_0.95"]} rows`}.</p>
          </Card>
        ))}
        {pendingAssistants.map((a) => (
          <Card key={a.id} hover onClick={() => navigate("assistants")} className="p-5">
            <div className="flex items-start justify-between"><div><p className="text-xs text-ink-400">Assistant</p><p className="font-display text-base font-bold text-ink-900">{a.name}</p></div><Badge status="pending" /></div>
            <p className="mt-3 text-sm text-ink-500">{a.faqs} FAQs from {a.dataset_id}, created by {a.created_by}.</p>
          </Card>
        ))}
      </div>
      {!pendingModels.length && !preparedData.length && !pendingAssistants.length && <Card><Empty icon="shield" title="Nothing waiting for approval" sub="New models and prepared datasets appear here." /></Card>}
    </div>
  );
}
