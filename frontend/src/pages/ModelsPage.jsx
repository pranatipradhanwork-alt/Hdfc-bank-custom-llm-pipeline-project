import { useState } from "react";
import { api, formatTime, useApi } from "../api";
import { Badge, Breadcrumb, Button, Card, Info, Loading, PageHeader, SectionTitle, Stages } from "../components/ui";

/* ============================== MODELS ============================== */
export function modelStatus(name, entry, registry) {
  return name === registry.live ? "live" : entry.status;
}

export function ModelsPage({ navigate }) {
  const models = useApi("/v1/models");
  if (!models.data) return <Loading error={models.error} />;
  const registry = models.data;

  return (
    <div className="fade-in">
      <PageHeader title="Model Registry" description="Every adapter version with its checksum, training run and evaluation. Only approved versions can go live." />
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3">
        {Object.entries(registry.versions).map(([name, m]) => (
          <Card key={name} hover onClick={() => navigate("modelDetail", name)} className="flex flex-col gap-4 p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="font-display text-base font-bold text-ink-900">{name}</p>
                <p className="text-xs text-ink-400">{m.base_model || "-"}</p>
                {m.training && m.training.platform && <p className="text-xs text-ink-400">Trained on {m.training.platform} · {m.training.device}</p>}
              </div>
              <Badge status={modelStatus(name, m, registry)} />
            </div>
            <div className="grid grid-cols-2 gap-3 border-t border-ink-100 pt-4 text-sm">
              <div><p className="text-xs text-ink-400">ROUGE-L (reworded)</p><p className="font-semibold tabular-nums text-ink-800">{m.evaluation.rougeL_finetuned_with_rag_reworded ?? "-"}</p></div>
              <div><p className="text-xs text-ink-400">Retrieval hit@3</p><p className="font-semibold tabular-nums text-ink-800">{m.evaluation.retrieval_hit_at_3 ?? "-"}</p></div>
            </div>
            <p className="border-t border-ink-100 pt-4 font-mono text-xs text-ink-400">sha256 {m.adapter_sha256.slice(0, 16)}…</p>
          </Card>
        ))}
      </div>
    </div>
  );
}

export function ModelDetailPage({ id, navigate, user, setToast }) {
  const models = useApi("/v1/models");
  const [busy, setBusy] = useState(false);
  if (!models.data) return <Loading error={models.error} />;
  const registry = models.data;
  const m = registry.versions[id];
  if (!m) return <Loading error={`Unknown model ${id}`} />;
  const status = modelStatus(id, m, registry);
  const stage = { pending: 2, rejected: 2, approved: 3, live: 4 }[status];
  const training = m.training || {};

  async function action(path, body, message) {
    setBusy(true);
    try {
      await api(path, "POST", body);
      setToast({ message });
      models.reload();
    } catch (err) {
      setToast({ message: err.message, type: "error" });
    }
    setBusy(false);
  }

  return (
    <div className="fade-in">
      <Breadcrumb items={[{ label: "Models", onClick: () => navigate("models") }, { label: id }]} />
      <PageHeader title={id} description={m.notes} action={
        <div className="flex flex-wrap items-center gap-2">
          {status === "pending" && user.permissions.includes("review_model") && <>
            <Button variant="success" size="sm" icon="check" disabled={busy} onClick={() => action(`/v1/models/${id}/review`, { status: "approved" }, `${id} approved`)}>Approve</Button>
            <Button variant="danger" size="sm" icon="x" disabled={busy} onClick={() => action(`/v1/models/${id}/review`, { status: "rejected" }, `${id} rejected`)}>Reject</Button>
          </>}
          {status === "approved" && user.permissions.includes("promote") &&
            <Button variant="primary" size="sm" icon="rocket" disabled={busy} onClick={() => action(`/v1/models/${id}/promote`, null, `${id} is now live`)}>{busy ? "Loading model…" : "Promote to production"}</Button>}
          <Badge status={status} />
        </div>} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="p-6">
          <SectionTitle>Model information</SectionTitle>
          <Info items={[["Base model", m.base_model || "-"], ["Training run", m.run_id], ["Training data", training.dataset_version != null ? `Delta v${training.dataset_version}` : "-"],
            ["Config", training.config || "-"], ["Trained on", training.platform ? `${training.platform} · ${training.device}` : "-"], ["Code commit", training.git_commit || "-"], ["MLflow run", training.mlflow_run_id ? training.mlflow_run_id.slice(0, 12) + "…" : "-"],
            ["Artifact", `HF repo folder ${m.hf_repo_folder}`], ["SHA-256", m.adapter_sha256]]} />
        </Card>
        <Card className="p-6">
          <SectionTitle>Evaluation</SectionTitle>
          <Info items={Object.entries(m.evaluation).map(([k, v]) => [k.replace(/_/g, " "), v == null ? "not measured" : String(v)])} />
        </Card>
        <Card className="p-6">
          <SectionTitle>Lifecycle</SectionTitle>
          <Stages stages={["Trained", "Evaluated", "Reviewed", "Approved", "Live"]} current={stage} failed={status === "rejected"} />
          <div className="mt-6 border-t border-ink-100 pt-4">
            <p className="mb-2 text-xs font-medium uppercase text-ink-400">History</p>
            {registry.history.filter((h) => h.to === id || h.from === id).reverse().map((h, i) => (
              <p key={i} className="py-1 text-sm text-ink-600"><span className="text-ink-400">{formatTime(h.time)}</span> · {h.action} {h.by ? `by ${h.by}` : ""}</p>
            ))}
            {!registry.history.some((h) => h.to === id || h.from === id) && <p className="text-sm text-ink-400">No changes recorded yet.</p>}
          </div>
        </Card>
      </div>
    </div>
  );
}
