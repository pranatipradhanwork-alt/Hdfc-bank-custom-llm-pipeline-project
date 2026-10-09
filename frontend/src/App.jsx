import { useState, useEffect } from "react";
import { api, getToken, setToken } from "./api";
import { Icon } from "./components/Icon";
import { Sidebar } from "./components/Sidebar";
import { Badge, Button, Loading, Toast } from "./components/ui";
import { NAV_ITEMS, PAGE_PARENT, ROLE_LABELS, pageFromUrl, pageUrl } from "./navigation";
import { ApprovalsPage } from "./pages/ApprovalsPage";
import { AssistantPage } from "./pages/AssistantPage";
import { AssistantsPage } from "./pages/AssistantsPage";
import { AuditLogsPage } from "./pages/AuditLogsPage";
import { DashboardPage } from "./pages/DashboardPage";
import { DatasetDetailPage, DatasetsPage } from "./pages/DatasetsPage";
import { DeploymentsPage } from "./pages/DeploymentsPage";
import { EvaluationsPage } from "./pages/EvaluationsPage";
import { LoginPage } from "./pages/LoginPage";
import { ModelDetailPage, ModelsPage } from "./pages/ModelsPage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { SettingsPage } from "./pages/SettingsPage";
import { TrainingJobsPage } from "./pages/TrainingJobsPage";
import { UsersPage } from "./pages/UsersPage";

/* App shell: session, routing, header and the current page */
export function App() {
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(true);
  // depth counts the pages opened since sign-in, so the Back button only shows when there is a page to go back to
  const [route, setRoute] = useState({ page: null, id: null, depth: 0 });
  const [toast, setToast] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);

  // On load, reuse a saved login if it is still valid
  useEffect(() => {
    if (!getToken()) { setChecking(false); return; }
    api("/v1/auth/me").then(startSession).catch(() => setToken(null)).finally(() => setChecking(false));
  }, []);

  useEffect(() => {
    const expired = () => { setToken(null); setUser(null); };
    window.addEventListener("session-expired", expired);
    return () => window.removeEventListener("session-expired", expired);
  }, []);

  // Each page is a browser history entry, so the browser's Back and Forward buttons move between pages
  useEffect(() => {
    const onPopState = (e) => {
      if (e.state && e.state.page) {
        setRoute(e.state);
        setMenuOpen(false);
        window.scrollTo({ top: 0 });
      }
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (e) => { if (e.key === "Escape") setMenuOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menuOpen]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 3500);
    return () => clearTimeout(timer);
  }, [toast]);

  function startSession(u) {
    setUser(u);
    const home = { page: u.permissions.includes("view_platform") ? "dashboard" : "assistant", id: null };
    const start = { ...(pageFromUrl(u) || home), depth: 0 };
    window.history.replaceState(start, "", pageUrl(start));
    setRoute(start);
  }

  async function logout() {
    try { await api("/v1/auth/logout", "POST"); } catch {}
    window.history.replaceState(null, "", window.location.pathname);
    setToken(null);
    setUser(null);
  }

  function navigate(page, id) {
    const next = { page, id: id || null, depth: route.depth + 1 };
    window.history.pushState(next, "", pageUrl(next));
    setRoute(next);
    setMenuOpen(false);
    window.scrollTo({ top: 0 });
  }

  if (checking) return <Loading />;
  if (!user) return <LoginPage onLogin={startSession} />;

  const props = { navigate, user, setToast, id: route.id };
  const pages = {
    dashboard: DashboardPage, assistant: AssistantPage, assistants: AssistantsPage, datasets: DatasetsPage, datasetDetail: DatasetDetailPage,
    training: TrainingJobsPage, models: ModelsPage, modelDetail: ModelDetailPage, evaluations: EvaluationsPage,
    approvals: ApprovalsPage, deployments: DeploymentsPage, monitoring: MonitoringPage, audit: AuditLogsPage, users: UsersPage,
  };
  const activeKey = PAGE_PARENT[route.page] || route.page;
  const label = (NAV_ITEMS.find((n) => n.key === activeKey) || {}).label;
  const Page = pages[route.page];

  return (
    <div className="min-h-screen bg-ink-50">
      <Sidebar page={activeKey} navigate={navigate} user={user} open={menuOpen} onClose={() => setMenuOpen(false)} />
      <div className="lg:pl-64">
        <header className="sticky top-0 z-20 flex h-16 items-center gap-2 border-b border-ink-100 bg-white/90 px-4 backdrop-blur sm:gap-4 sm:px-6">
          <button onClick={() => setMenuOpen(true)} className="-ml-1 rounded-lg p-1.5 text-ink-500 hover:bg-ink-100 lg:hidden" aria-label="Open menu"><Icon name="menu" size={20} /></button>
          {route.depth > 0 && <Button variant="ghost" size="sm" icon="arrowLeft" onClick={() => window.history.back()}>Back</Button>}
          <p className="min-w-0 truncate font-display text-sm font-semibold text-ink-700">{label}</p>
          <span className="ml-auto hidden sm:block"><Badge status="live">{ROLE_LABELS[user.role]}</Badge></span>
          <Button variant="ghost" size="sm" icon="logout" onClick={logout} className="ml-auto sm:ml-0"><span className="hidden sm:inline">Sign out</span></Button>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
          {user.role === "reviewer" && (
            <div className="mb-5 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <Icon name="shield" size={16} className="mt-0.5 shrink-0" />
              <span><b>Read-only reviewer access.</b> You can open every page and ask the assistants. Approving, training, deploying and other changes are disabled; they are shown in the demo video.</span>
            </div>
          )}
          {route.page === "settings" ? <SettingsPage user={user} onLogout={logout} /> : Page ? <Page key={route.page + (route.id || "")} {...props} /> : null}
        </main>
      </div>
      <Toast toast={toast} />
    </div>
  );
}
