import { useState } from "react";
import { api, formatTime, useApi } from "../api";
import { Badge, Button, Card, CommandHint, Field, Info, Loading, Modal, PageHeader, Table, inputClass } from "../components/ui";

/* ============================== TRAINING JOBS ============================== */
export function NewRunModal({ onClose, onDone, setToast }) {
  const datasets = useApi("/v1/datasets");
  const options = useApi("/v1/base-models");
  const [form, setForm] = useState({ name: "", dataset_id: "", base_model: "", config: "configs/training/cuda-qlora.yaml", seed: 42 });
  const set = (key) => (e) => setForm({ ...form, [key]: e.target.value });
  if (!datasets.data || !options.data) return <Modal title="New training job" onClose={onClose}><Loading error={datasets.error || options.error} /></Modal>;
  const approved = datasets.data.filter((d) => d.status === "approved");

  async function submit(e) {
    e.preventDefault();
    try {
      const run = await api("/v1/runs", "POST", { ...form, seed: Number(form.seed),
        dataset_id: form.dataset_id || (approved[0] && approved[0].id), base_model: form.base_model || options.data.approved_base_models[0] });
      setToast({ message: `Run ${run.id} queued for the GPU worker` });
      onDone();
    } catch (err) {
      setToast({ message: err.message, type: "error" });
    }
  }

  return (
    <Modal title="New training job" onClose={onClose}>
      <form onSubmit={submit} className="space-y-3">
        <Field label="Run name" hint="Lowercase letters, numbers and _ only, e.g. llama_v3. Also the adapter folder name."><input className={inputClass} value={form.name} onChange={set("name")} required pattern="[a-z0-9_]+" /></Field>
        <Field label="Approved dataset">
          <select className={inputClass} value={form.dataset_id} onChange={set("dataset_id")}>
            {approved.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.id}, Delta v{d.delta_version})</option>)}
          </select>
        </Field>
        <Field label="Approved base model">
          <select className={inputClass} value={form.base_model} onChange={set("base_model")}>
            {options.data.approved_base_models.map((m) => <option key={m}>{m}</option>)}
          </select>
        </Field>
        <Field label="LoRA / QLoRA config">
          <select className={inputClass} value={form.config} onChange={set("config")}>
            {options.data.configs.map((c) => <option key={c}>{c}</option>)}
          </select>
        </Field>
        <Field label="Seed"><input className={inputClass} type="number" value={form.seed} onChange={set("seed")} /></Field>
        <div className="flex justify-end gap-2 pt-2">
          <Button onClick={onClose}>Cancel</Button>
          <Button type="submit" variant="primary" disabled={!approved.length}>Queue run</Button>
        </div>
      </form>
    </Modal>
  );
}

export const HARDWARE_LABELS = { "CUDA (Quantized)": "NVIDIA GPU (QLoRA)", "MPS": "Apple GPU (MPS)", "CPU": "CPU" };

export function TrainingJobsPage({ user, setToast }) {
  const runs = useApi("/v1/runs");
  const [showForm, setShowForm] = useState(false);
  const [open, setOpen] = useState(null);
  if (!runs.data) return <Loading error={runs.error} />;
  const rows = [...runs.data].reverse();

  return (
    <div className="fade-in">
      <PageHeader title="Training Jobs" description="LoRA / QLoRA fine-tuning runs. They were run on a Windows NVIDIA GPU machine and on a Mac (Apple GPU); every run records its base model, data version, config, seed and code commit."
        action={user.permissions.includes("request_run") && <Button variant="primary" icon="plus" onClick={() => setShowForm(true)}>New Training Job</Button>} />
      <Card>
        <Table rows={rows} onRowClick={setOpen} columns={[
          { label: "Run", render: (r) => <span className="font-medium text-ink-900">{r.id}</span> },
          { label: "Base model", render: (r) => r.base_model },
          { label: "Platform", render: (r) => r.platform ? `${r.platform} · ${HARDWARE_LABELS[r.device] || r.device}` : "-" },
          { label: "Data", render: (r) => `Delta v${r.dataset_version}` },
          { label: "Status", render: (r) => <Badge status={r.status} /> },
          { label: "Steps", render: (r) => r.max_steps === -1 ? `full (${r.epochs ?? 3} epoch${(r.epochs ?? 3) === 1 ? "" : "s"})` : r.max_steps ?? "-" },
          { label: "Test loss", render: (r) => r.test_loss == null ? "-" : r.rag_training ? `${r.test_loss} (with FAQs in prompt)` : r.test_loss },
          { label: "Duration", render: (r) => r.minutes != null ? `${r.minutes} min` : "-" },
          { label: "Started", render: (r) => formatTime(r.started_at || r.requested_at) },
        ]} />
      </Card>
      {open && (
        <Modal title={`Run manifest: ${open.id}`} onClose={() => setOpen(null)}>
          <Info items={Object.entries(open).filter(([k]) => k !== "lora" && k !== "command").map(([k, v]) => [k.replace(/_/g, " "), String(v ?? "-")])} />
          {open.lora && <div className="mt-4 border-t border-ink-100 pt-4"><p className="mb-2 text-xs font-medium uppercase text-ink-400">LoRA</p>
            <Info items={Object.entries(open.lora).map(([k, v]) => [k, Array.isArray(v) ? v.join(", ") : String(v)])} /></div>}
          {open.status === "queued" && <CommandHint command={open.command} />}
        </Modal>
      )}
      {showForm && <NewRunModal setToast={setToast} onClose={() => setShowForm(false)} onDone={() => { setShowForm(false); runs.reload(); }} />}
    </div>
  );
}
