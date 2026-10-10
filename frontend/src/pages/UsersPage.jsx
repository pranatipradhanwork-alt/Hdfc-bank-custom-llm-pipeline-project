import { api, useApi } from "../api";
import { Card, Loading, PageHeader, Table } from "../components/ui";

/* ============================== USERS (admin) ============================== */
export function UsersPage({ setToast }) {
  const users = useApi("/v1/users");
  const assistants = useApi("/v1/assistants");
  if (!users.data || !assistants.data) return <Loading error={users.error || assistants.error} />;

  async function toggle(user, assistantId) {
    const next = user.assistants.includes(assistantId) ? user.assistants.filter((a) => a !== assistantId) : [...user.assistants, assistantId];
    try {
      await api(`/v1/users/${user.username}/assistants`, "POST", { assistants: next });
      setToast({ message: `Updated assistants for ${user.username}` });
      users.reload();
    } catch (err) {
      setToast({ message: err.message, type: "error" });
    }
  }

  return (
    <div className="fade-in">
      <PageHeader title="Users & Access" description="Roles decide what each person can do. Employees only see the assistants assigned to them." />
      <Card>
        <Table rows={users.data} columns={[
          { label: "User", render: (u) => <div><p className="font-medium text-ink-900">{u.name}</p><p className="text-xs text-ink-400">{u.username}</p></div> },
          { label: "Role", render: (u) => u.role.replace("_", " ") },
          { label: "Team", render: (u) => u.team },
          { label: "Assigned assistants", render: (u) => (
            <div className="flex flex-wrap gap-2">
              {assistants.data.filter((a) => a.status === "approved").map((a) => (
                <label key={a.id} className="flex items-center gap-1.5 text-xs">
                  <input type="checkbox" checked={u.assistants.includes(a.id)} onChange={() => toggle(u, a.id)} /> {a.name}
                </label>
              ))}
            </div>) },
          { label: "Permissions", render: (u) => <span className="text-xs text-ink-400">{u.permissions.length} actions</span> },
        ]} />
      </Card>
    </div>
  );
}
