import { Icon } from "./Icon";
import { NAV_ITEMS, ROLE_LABELS } from "../navigation";

/* Sidebar with the pages this user's role may open */
export function initials(name) {
  return name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase();
}

export function Sidebar({ page, navigate, user, open, onClose }) {
  // Always shown on wide screens; on narrow screens it slides in over the page from the menu button
  return (
    <>
    {open && <div className="fixed inset-0 z-30 bg-ink-900/40 lg:hidden" onClick={onClose} />}
    <aside className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-ink-100 bg-white transition-transform duration-200 lg:translate-x-0 ${open ? "translate-x-0" : "-translate-x-full"}`}>
      <div className="flex items-center gap-2.5 border-b border-ink-100 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 font-display text-sm font-extrabold text-white">H</div>
        <div className="leading-tight">
          <p className="font-display text-[14px] font-bold text-ink-900">HDFC AI Platform</p>
          <p className="text-[11px] font-medium text-ink-400">Enterprise AI Management</p>
        </div>
        <button onClick={onClose} className="ml-auto text-ink-400 hover:text-ink-700 lg:hidden" aria-label="Close menu"><Icon name="x" /></button>
      </div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 py-4">
        {NAV_ITEMS.filter((item) => user.permissions.includes(item.needs)).map((item) => {
          const active = page === item.key;
          return (
            <button key={item.key} onClick={() => navigate(item.key)}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors duration-150 ${active ? "bg-brand-50 text-brand-700" : "text-ink-600 hover:bg-ink-50 hover:text-ink-900"}`}>
              <Icon name={item.icon} size={17} className={active ? "text-brand-600" : "text-ink-400"} />
              {item.label}
              {active && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-brand-600" />}
            </button>
          );
        })}
      </nav>
      <div className="border-t border-ink-100 p-4">
        <div className="flex items-center gap-3 rounded-lg px-1.5 py-1.5">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-ink-800 text-xs font-bold text-white">{initials(user.name)}</div>
          <div className="min-w-0 flex-1 leading-tight">
            <p className="truncate text-sm font-semibold text-ink-900">{user.name}</p>
            <p className="truncate text-xs text-ink-400">{ROLE_LABELS[user.role]}</p>
          </div>
        </div>
      </div>
    </aside>
    </>
  );
}
