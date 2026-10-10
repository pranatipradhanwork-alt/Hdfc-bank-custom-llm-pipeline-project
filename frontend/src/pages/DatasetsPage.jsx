import { useState } from "react";
import { api, formatTime, useApi } from "../api";
import { Icon } from "../components/Icon";
import { Badge, Breadcrumb, Button, Card, CommandHint, Empty, Field, Info, Loading, Modal, PageHeader, SectionTitle, Stages, Table, inputClass } from "../components/ui";

/* ============================== DATASETS ============================== */
export const DATASET_STAGES = ["Registered", "Preparing", "Prepared", "Approved"];

export function RegisterDatasetModal({ onClose, onDone, setToast, datasets }) {
  const [form, setForm] = useState({ name: "", source: "", owner: "", purpose: "", classification: "internal", permission_basis: "", retention: "", parent_id: "", keywords: "" });
  const set = (key) => (e) => setForm({ ...form, [key]: e.target.value });
  const approved = datasets.filter((d) => d.status === "approved" && !d.parent_id);

  async function submit(e) {
    e.preventDefault();
    const body = { ...form, parent_id: form.parent_id || null,
      keywords: form.keywords.split(",").map((k) => k.trim()).filter(Boolean) };
    try {
      const d = await api("/v1/datasets", "POST", body);
      setToast({ message: `Registered ${d.id} (${d.name})` });
      onDone();
    } catch (err) {
      setToast({ message: err.message, type: "error" });
    }
  }

  return (
    <Modal title="Register dataset" onClose={onClose}>
      <form onSubmit={submit} className="space-y-3">
        <Field label="Name"><input className={inputClass} value={form.name} onChange={set("name")} required /></Field>
        <Field label="Source" hint="Where the data comes from, e.g. a file path or system of record"><input className={inputClass} value={form.source} onChange={set("source")} required /></Field>
        <Field label="Data owner" hint="The team accountable for this data, e.g. fd-team"><input className={inputClass} value={form.owner} onChange={set("owner")} required /></Field>
        <Field label="Purpose" hint="Lowercase with _, e.g. fd_assistant"><input className={inputClass} value={form.purpose} onChange={set("purpose")} required pattern="[a-z0-9_]+" /></Field>
        <Field label="Team dataset: take FAQs from" hint="Optional. Pick an approved dataset to select this team's FAQs from.">
          <select className={inputClass} value={form.parent_id} onChange={set("parent_id")}>
            <option value="">No, this is a new data source</option>
            {approved.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.id}, Delta v{d.delta_version})</option>)}
          </select>
        </Field>
        {form.parent_id && <Field label="Keywords" hint="Comma separated. FAQs mentioning any of these are included, e.g. fixed deposit, fd, recurring deposit">
          <input className={inputClass} value={form.keywords} onChange={set("keywords")} required /></Field>}
        <Field label="Classification" hint="Restricted data is refused for fine-tuning">
          <select className={inputClass} value={form.classification} onChange={set("classification")}>
            {["public", "internal", "confidential", "restricted"].map((c) => <option key={c}>{c}</option>)}
          </select>
        </Field>
        <Field label="Permission basis" hint="Why the bank may use this data for this purpose"><input className={inputClass} value={form.permission_basis} onChange={set("permission_basis")} required /></Field>
        <Field label="Retention"><input className={inputClass} value={form.retention} onChange={set("retention")} required /></Field>
        <div className="flex justify-end gap-2 pt-2">
          <Button onClick={onClose}>Cancel</Button>
          <Button type="submit" variant="primary">Register</Button>
        </div>
      </form>
    </Modal>
  );
}

export function DatasetsPage({ navigate, user, setToast }) {
  const datasets = useApi("/v1/datasets");
  const [showForm, setShowForm] = useState(false);
  if (!datasets.data) return <Loading error={datasets.error} />;
  const canRegister = user.permissions.includes("register_dataset");

  return (
    <div className="fade-in">
      <PageHeader title="Datasets" description="Registered data sources. Only approved datasets can be used for fine-tuning."
        action={canRegister && <Button variant="primary" icon="plus" onClick={() => setShowForm(true)}>Register Dataset</Button>} />
      <Card>
        <Table rows={datasets.data} onRowClick={(d) => navigate("datasetDetail", d.id)} columns={[
          { label: "Dataset", render: (d) => (
            <div className="flex items-center gap-3">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-50 text-ink-500"><Icon name="database" size={15} /></span>
              <div><p className="font-medium text-ink-900">{d.name}</p><p className="text-xs text-ink-400">{d.id}</p></div>
            </div>) },
          { label: "Delta version", render: (d) => d.delta_version != null ? `v${d.delta_version}` : "-" },
          { label: "Owner", render: (d) => d.owner },
          { label: "Classification", render: (d) => d.classification },
          { label: "Status", render: (d) => <Badge status={d.status} /> },
          { label: "Rows", render: (d) => d.quality_report ? d.quality_report.final_rows.toLocaleString("en-IN") : "-" },
          { label: "Last change", render: (d) => formatTime(d.history[d.history.length - 1].time) },
        ]} />
      </Card>
      {showForm && <RegisterDatasetModal datasets={datasets.data} setToast={setToast} onClose={() => setShowForm(false)} onDone={() => { setShowForm(false); datasets.reload(); }} />}
    </div>
  );
}

export function DatasetDetailPage({ id, navigate, user, setToast }) {
  const dataset = useApi(`/v1/datasets/${id}`);
  if (!dataset.data) return <Loading error={dataset.error} />;
  const d = dataset.data;
  const q = d.quality_report;
  const stage = DATASET_STAGES.map((s) => s.toLowerCase()).indexOf(d.status);

  async function action(path, message) {
    try {
      await api(path, "POST");
      setToast({ message });
      dataset.reload();
    } catch (err) {
      setToast({ message: err.message, type: "error" });
    }
  }

  return (
    <div className="fade-in">
      <Breadcrumb items={[{ label: "Datasets", onClick: () => navigate("datasets") }, { label: d.name }]} />
      <PageHeader title={d.name} description={d.source} action={
        <div className="flex gap-2">
          {d.status === "registered" && user.permissions.includes("prepare_dataset") && (d.parent_id
            ? <Button variant="primary" onClick={() => action(`/v1/datasets/${d.id}/prepare`, "Knowledge index built")}>Build knowledge index</Button>
            : <Button variant="primary" onClick={() => action(`/v1/datasets/${d.id}/prepare`, "Preparation requested")}>Request preparation</Button>)}
          {d.status === "prepared" && user.permissions.includes("approve_dataset") &&
            <Button variant="success" icon="check" onClick={() => action(`/v1/datasets/${d.id}/approve`, `${d.name} approved and frozen`)}>Approve & freeze</Button>}
          <Badge status={d.status} />
        </div>} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="p-6 lg:col-span-2">
          <SectionTitle>Governance</SectionTitle>
          <Info items={[["ID", d.id], ["Owner", d.owner], ["Purpose", d.purpose], ["Classification", d.classification],
            ...(d.parent_id ? [["Selected from", d.parent_id], ["Keywords", d.keywords.join(", ")]] : []),
            ["Permission basis", d.permission_basis], ["Retention", d.retention], ["Delta version", d.delta_version ?? "-"], ["Approved by", d.approved_by || "-"]]} />
          <div className="mt-6 border-t border-ink-100 pt-6">
            <SectionTitle>Dataset lifecycle</SectionTitle>
            <Stages stages={DATASET_STAGES} current={d.status === "approved" ? 4 : stage} />
            {d.status === "preparing" && <CommandHint command={`python clean_data.py && python control_plane.py --record-prepared ${d.id}`} />}
          </div>
          <div className="mt-6 border-t border-ink-100 pt-6">
            <SectionTitle>History</SectionTitle>
            {d.history.map((h, i) => (
              <div key={i} className="flex items-start gap-3 py-1.5 text-sm">
                <span className="w-32 shrink-0 text-ink-400">{formatTime(h.time)}</span>
                <Badge status={h.status} />
                <span className="text-ink-600">{h.note}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card className="p-6">
          <SectionTitle>Quality report</SectionTitle>
          {q && d.parent_id ? <Info items={[
            ["FAQs selected", q.final_rows],
            ["Selected from", `${q.derived_from} (Delta v${q.delta_version})`],
            ["Keywords", q.keywords.join(", ")],
            ["PII masking", "Done upstream in the parent dataset"],
            ["Duplicates", "Removed upstream in the parent dataset"],
            ["Minimum for approval", "10 FAQs"],
          ]} /> : q ? <Info items={[
            ["Raw rows", q.raw_rows.toLocaleString("en-IN")],
            ["Final rows", q.final_rows.toLocaleString("en-IN")],
            ["Exact duplicates removed", q.exact_duplicates_removed],
            ["Near duplicates removed", q.near_duplicates_removed],
            ["PII values masked", Object.values(q.pii_masks_applied).reduce((a, b) => a + b, 0)],
            ["Train / val / test", `${q.split_rows.train} / ${q.split_rows.validation} / ${q.split_rows.test}`],
            ["Leaked test rows (≥0.95)", q.contamination["test_rows_at_or_above_0.95"]],
            ["Max test-train similarity", q.contamination.max_similarity_to_train],
          ]} /> : <Empty icon="clipboard" title="Not prepared yet" sub="The quality report appears after preparation." />}
          {q && !d.parent_id && <div className="mt-5 border-t border-ink-100 pt-4">
            <Badge status={q.contamination["test_rows_at_or_above_0.95"] === 0 ? "passed" : "failed"}>Leakage check {q.contamination["test_rows_at_or_above_0.95"] === 0 ? "passed" : "failed"}</Badge>
          </div>}
          {q && d.parent_id && <div className="mt-5 border-t border-ink-100 pt-4">
            <Badge status={q.final_rows >= 10 ? "passed" : "failed"}>Size check {q.final_rows >= 10 ? "passed" : "failed"}</Badge>
          </div>}
        </Card>
      </div>
    </div>
  );
}
