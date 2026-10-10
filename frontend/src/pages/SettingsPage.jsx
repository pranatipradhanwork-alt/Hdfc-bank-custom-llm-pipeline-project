import { initials } from "../components/Sidebar";
import { Button, Card, Info, PageHeader, SectionTitle } from "../components/ui";
import { ROLE_LABELS } from "../navigation";

/* ============================== SETTINGS ============================== */
export function SettingsPage({ user, onLogout }) {
  return (
    <div className="fade-in max-w-3xl">
      <PageHeader title="Settings" description="Your profile and what your role allows." />
      <Card className="p-6">
        <SectionTitle action={<Button variant="danger" size="sm" icon="logout" onClick={onLogout}>Sign out</Button>}>Profile</SectionTitle>
        <div className="flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-ink-800 text-base font-bold text-white">{initials(user.name)}</div>
          <div><p className="font-semibold text-ink-900">{user.name}</p><p className="text-sm text-ink-500">{ROLE_LABELS[user.role]} · {user.team}</p></div>
        </div>
        <div className="mt-5 border-t border-ink-100 pt-5">
          <Info items={[["Username", user.username], ["Role", ROLE_LABELS[user.role]], ["Team", user.team || "-"], ["Assigned assistants", user.assistants.join(", ") || "none"]]} />
        </div>
      </Card>
      <Card className="mt-6 p-6">
        <SectionTitle>Platform access</SectionTitle>
        <div className="flex flex-wrap gap-2">
          {user.permissions.map((p) => <span key={p} className="rounded-md bg-ink-100 px-2 py-1 text-xs font-medium text-ink-600">{p.replace(/_/g, " ")}</span>)}
        </div>
      </Card>
    </div>
  );
}
