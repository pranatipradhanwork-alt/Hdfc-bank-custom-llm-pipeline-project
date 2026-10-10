import { ResponsiveContainer, BarChart, Bar, Legend, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { useApi } from "../api";
import { Card, Info, Loading, PageHeader, SectionTitle, Table } from "../components/ui";

/* ============================== EVALUATIONS ============================== */
export function EvaluationsPage() {
  const models = useApi("/v1/models");
  if (!models.data) return <Loading error={models.error} />;
  const rows = Object.entries(models.data.versions).map(([name, m]) => ({ name, platform: (m.training || {}).platform, ...m.evaluation }));
  const chart = rows.filter((r) => r.rougeL_finetuned_with_rag != null).map((r) => ({ name: r.name, platform: r.platform === "macOS" ? "Mac" : r.platform, version: r.dataset_same_as ?? r.dataset_version, test: r.rougeL_finetuned_with_rag, reworded: r.rougeL_finetuned_with_rag_reworded }));

  return (
    <div className="fade-in">
      <PageHeader title="Model Evaluation" description="Quality before release. Base vs fine-tuned and with vs without retrieval are compared in evaluate.py; the release comparison is shown here. Models were trained and tested on both a Windows NVIDIA GPU machine and a Mac." />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="p-6 lg:col-span-2">
          <SectionTitle>ROUGE-L by model (fine-tuned + retrieval)</SectionTitle>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chart} margin={{ top: 8, right: 12, left: 8, bottom: 16 }} barGap={4} barCategoryGap="28%">
                <CartesianGrid vertical={false} stroke="#EEEEF1" />
                <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} height={44}
                  label={{ value: "Model (trained on · data version)", position: "insideBottom", offset: -12, fontSize: 11, fill: "#5C5C66" }}
                  tick={({ x, y, payload, index }) => (
                  <text x={x} y={y + 12} textAnchor="middle" fontSize={10} fill="#8C8C99">
                    <tspan x={x} fill="#3A3A42">{payload.value}</tspan>
                    <tspan x={x} dy="1.2em">{chart[index] && `${chart[index].platform || "-"} · data v${chart[index].version}`}</tspan>
                  </text>)} />
                <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} width={36}
                  label={{ value: "ROUGE-L (0-1, higher is better)", angle: -90, position: "insideLeft", offset: -2, dy: 90, fontSize: 11, fill: "#5C5C66" }} />
                <Tooltip contentStyle={{ fontSize: 12 }} cursor={{ fill: "#F5F5F7" }} formatter={(v) => (v == null ? "not measured" : v)} />
                <Legend verticalAlign="top" align="right" iconType="square" iconSize={10} wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
                <Bar dataKey="test" name="Held-out test" fill="#AD1730" radius={[3, 3, 0, 0]} maxBarSize={36} />
                <Bar dataKey="reworded" name="Reworded questions" fill="#0F9B8E" radius={[3, 3, 0, 0]} maxBarSize={36} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-3 text-xs text-ink-400">Each model is scored on its own data version, so bars are only directly comparable at the same version. The Mac runs use a local copy of v7 (same FAQs and split). Quote the reworded score: the held-out score is inflated because the model saw near-copies of those answers during training. A missing bar means not measured yet.</p>
        </Card>
        <Card className="p-6">
          <SectionTitle>What is measured</SectionTitle>
          <Info items={[["ROUGE-L", "Overlap with the official FAQ answer (0-1)"], ["Retrieval hit@3", "Right FAQ in the top 3 results"],
            ["Unsupported figures", "Answers stating amounts not in the cited FAQs"], ["Safety", "Guardrail tests (pytest) and promptfoo red-team suite"],
            ["Human review", "People grade 50 served answers as correct, partly or wrong, and flag harmful ones"]]} />
        </Card>
      </div>
      <Card className="mt-6">
        <Table rows={rows} columns={[
          { label: "Version", render: (r) => <span className="font-medium text-ink-900">{r.name}</span> },
          { label: "Trained on", render: (r) => r.platform || "-" },
          { label: "Eval data", render: (r) => r.dataset_same_as ? `Delta v${r.dataset_same_as} (Mac copy)` : `Delta v${r.dataset_version}` },
          { label: "ROUGE-L (test)", render: (r) => r.rougeL_finetuned_with_rag ?? "-" },
          { label: "ROUGE-L (reworded)", render: (r) => r.rougeL_finetuned_with_rag_reworded ?? "-" },
          { label: "Retrieval hit@3", render: (r) => r.retrieval_hit_at_3 ?? "-" },
          { label: "Unsupported figures", render: (r) => r.answers_with_unsupported_figures ?? "-" },
          { label: "Safety (promptfoo)", render: (r) => r.safety_promptfoo || "-" },
          { label: "Human review", render: (r) => r.human_review || "-" },
          { label: "Report", render: (r) => <span className="font-mono text-xs">{r.report || "-"}</span> },
        ]} />
      </Card>
    </div>
  );
}
