import { Link, NavLink, Outlet, Route, Routes } from "react-router-dom";
import { AskPage } from "./pages/AskPage";
import { HistoryPage } from "./pages/HistoryPage";
import { RunPage } from "./pages/RunPage";
import { StatsPage } from "./pages/StatsPage";

function Layout() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="container topbar-inner">
          <Link to="/" className="brand">
            <span className="brand-mark" aria-hidden="true" />
            council of models
          </Link>
          <nav className="nav" aria-label="Main">
            <NavLink to="/" end>
              Ask
            </NavLink>
            <NavLink to="/history">History</NavLink>
            <NavLink to="/stats">Stats</NavLink>
          </nav>
        </div>
      </header>
      <main className="container main">
        <Outlet />
      </main>
      <footer className="footer">
        <div className="container">
          Inspired by Karpathy&apos;s{" "}
          <a href="https://github.com/karpathy/llm-council" target="_blank" rel="noreferrer noopener">
            llm-council
          </a>
        </div>
      </footer>
    </div>
  );
}

function NotFound() {
  return (
    <section className="page">
      <h1>Not found</h1>
      <p>
        Nothing here. <Link to="/">Ask the council</Link> instead.
      </p>
    </section>
  );
}

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<AskPage />} />
        <Route path="runs/:id" element={<RunPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="stats" element={<StatsPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
