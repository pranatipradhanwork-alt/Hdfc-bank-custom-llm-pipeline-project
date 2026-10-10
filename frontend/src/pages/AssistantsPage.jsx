import { useState } from "react";
import { api, useApi } from "../api";
import { Badge, Button, Card, Field, Loading, Modal, PageHeader, inputClass } from "../components/ui";

/* ============================== ASSISTANTS ============================== */
export function CreateAssistantModal({ onClose, onDone, setToast }) {
  const datasets = useApi("/v1/datasets");
  const [form, setForm] = useState({ id: "", name: "", description: "", dataset_id: "" });
  const set = (key) => (e) => setForm({ ...form, [key]: e.target.value });
  if (!datasets.data) return <Modal title="Create assistant" onClose={onClose}><Loading error={datasets.error} /></Modal>;
  const approved = datasets.data.filter((d) => d.status === "approved");

  async function submit(e) {
    e.preventDefault();
    try {
      const a = await api("/v1/assistants", "POST", { ...form, dataset_id: form.dataset_id || (approved[0] && approved[0].id) });
      setToast({ message: `${a.name} created. It needs admin approval before anyone can use it.` });
      onDone();
    } catch (err) {
      setToast({ message: err.message, type: "error" });
    }
  }

  return (
    <Modal title="Create assistant" onClose={onClose}>
      <form onSubmit={submit} className="space-y-3">
        <Field label="Assistant id" hint="Lowercase with _, e.g. fd_assistant"><input className={inputClass} value={form.id} onChange={set("id")} required pattern="[a-z0-9_]+" /></Field>
        <Field label="Name"><input className={inputClass} value={form.name} onChange={set("name")} required placeholder="Fixed Deposit Assistant" /></Field>
        <Field label="What it helps with"><input className={inputClass} value={form.description} onChange={set("description")} placeholder="Answers fixed deposit questions for the FD team" /></Field>
        <Field label="Knowledge (approved dataset)" hint="The assistant answers only from this dataset's FAQs, using the shared fine-tuned model.">
          <select className={inputClass} value={form.dataset_id} onChange={set("dataset_id")}>
            {approved.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.id}, {d.quality_report.final_rows} FAQs)</option>)}
          </select>
        </Field>
        <div className="flex justify-end gap-2 pt-2">
          <Button onClick={onClose}>Cancel</Button>
          <Button type="submit" variant="primary" disabled={!approved.length}>Create</Button>
        </div>
      </form>
    </Modal>
  );
}

export function AssistantsPage({ user, setToast }) {
  const assistants = useApi("/v1/assistants");
  const [showForm, setShowForm] = useState(false);
  if (!assistants.data) return <Loading error={assistants.error} />;

  async function review(id, status) {
    try {
      await api(`/v1/assistants/${id}/review`, "POST", { status });
      setToast({ message: `Assistant ${status}` });
      assistants.reload();
    } catch (err) {
      setToast({ message: err.message, type: "error" });
    }
  }

  return (
    <div className="fade-in">
      <PageHeader title="Assistants" description="Each assistant is the shared fine-tuned model plus one team's approved knowledge. New assistants need admin approval before they can be assigned to employees."
        action={user.permissions.includes("create_assistant") && <Button variant="primary" icon="plus" onClick={() => setShowForm(true)}>Create Assistant</Button>} />
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3">
        {assistants.data.map((a) => (
          <Card key={a.id} className="flex flex-col gap-4 p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="font-display text-base font-bold text-ink-900">{a.name}</p>
                <p className="font-mono text-xs text-ink-400">{a.id}</p>
              </div>
              <Badge status={a.status} />
            </div>
            <p className="text-sm text-ink-500">{a.description || "-"}</p>
            <div className="grid grid-cols-2 gap-3 border-t border-ink-100 pt-4 text-sm">
              <div><p className="text-xs text-ink-400">Knowledge</p><p className="font-semibold text-ink-800">{a.faqs} FAQs · {a.dataset_id}</p></div>
              <div><p className="text-xs text-ink-400">Model</p><p className="font-semibold text-ink-800">{a.model_version}</p></div>
              <div><p className="text-xs text-ink-400">Owner</p><p className="font-semibold text-ink-800">{a.owner}</p></div>
              <div><p className="text-xs text-ink-400">Created / approved by</p><p className="font-semibold text-ink-800">{a.created_by} / {a.approved_by || "-"}</p></div>
            </div>
            {a.status === "pending" && user.permissions.includes("review_assistant") && (
              <div className="flex gap-2 border-t border-ink-100 pt-4">
                <Button variant="success" size="sm" icon="check" onClick={() => review(a.id, "approved")}>Approve</Button>
                <Button variant="danger" size="sm" icon="x" onClick={() => review(a.id, "rejected")}>Reject</Button>
              </div>
            )}
            {a.status === "pending" && !user.permissions.includes("review_assistant") &&
              <p className="border-t border-ink-100 pt-4 text-xs text-ink-400">Waiting for admin approval.</p>}
          </Card>
        ))}
      </div>
      {showForm && <CreateAssistantModal setToast={setToast} onClose={() => setShowForm(false)} onDone={() => { setShowForm(false); assistants.reload(); }} />}
    </div>
  );
}
