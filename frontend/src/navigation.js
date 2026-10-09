/* Navigation: role labels, the pages each role sees, and the URL <-> page mapping */
export const ROLE_LABELS = { admin: "Platform Admin", ai_engineer: "AI Platform Engineer", reviewer: "Reviewer (read-only)", employee: "Employee" };

// Which pages each role sees in the sidebar (the server enforces the same rules)
export const NAV_ITEMS = [
  { key: "dashboard", label: "Dashboard", icon: "dashboard", needs: "view_platform" },
  { key: "assistant", label: "AI Assistant", icon: "chat", needs: "use_assistant" },
  { key: "assistants", label: "Assistants", icon: "users", needs: "view_platform" },
  { key: "datasets", label: "Datasets", icon: "database", needs: "view_platform" },
  { key: "training", label: "Training Jobs", icon: "flask", needs: "view_platform" },
  { key: "models", label: "Models", icon: "box", needs: "view_platform" },
  { key: "evaluations", label: "Evaluations", icon: "clipboard", needs: "view_platform" },
  { key: "approvals", label: "Approvals", icon: "shield", needs: "view_platform" },
  { key: "deployments", label: "Deployments", icon: "rocket", needs: "view_platform" },
  { key: "monitoring", label: "Monitoring", icon: "activity", needs: "view_platform" },
  { key: "audit", label: "Audit Logs", icon: "file", needs: "view_platform" },
  { key: "users", label: "Users & Access", icon: "users", needs: "manage_users" },
  { key: "settings", label: "Settings", icon: "settings", needs: "use_assistant" },
];
export const PAGE_PARENT = { datasetDetail: "datasets", modelDetail: "models" };

export function pageUrl({ page, id }) {
  return "#" + page + (id ? "/" + encodeURIComponent(id) : "");
}

// The page in the address bar, if this user may open it
export function pageFromUrl(user) {
  const [page, id] = window.location.hash.slice(1).split("/");
  const item = NAV_ITEMS.find((n) => n.key === (PAGE_PARENT[page] || page));
  if (!item || !user.permissions.includes(item.needs)) return null;
  return { page, id: id ? decodeURIComponent(id) : null };
}
