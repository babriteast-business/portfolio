// Shared behaviour for every page: nav, reveals, copy buttons, blog index helpers.
window.REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

(function nav() {
  const nav = document.getElementById("nav");
  if (!nav) return;
  const set = () => nav.classList.toggle("solid", window.scrollY > 24 || document.body.dataset.solidNav === "1");
  set();
  window.addEventListener("scroll", set, { passive: true });
})();

window.revealAll = function (root) {
  const els = (root || document).querySelectorAll(".reveal:not(.shown)");
  if (!window.gsap || window.REDUCED) { els.forEach((e) => { e.classList.add("shown"); e.style.opacity = 1; e.style.transform = "none"; }); return; }
  gsap.registerPlugin(ScrollTrigger);
  ScrollTrigger.batch(els, {
    start: "top 88%",
    once: true,
    onEnter: (batch) => {
      batch.forEach((e) => e.classList.add("shown"));
      gsap.to(batch, { opacity: 1, y: 0, duration: 0.7, ease: "power3.out", stagger: 0.06, overwrite: true });
    },
  });
};

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (!b) return;
  navigator.clipboard?.writeText(b.dataset.copy).then(() => {
    const t = b.textContent; b.textContent = "Copied";
    setTimeout(() => (b.textContent = t), 1400);
  });
});

window.Blog = {
  async index() {
    const r = await fetch("posts/index.json", { cache: "no-cache" });
    if (!r.ok) throw new Error("Couldn't load posts/index.json");
    const all = await r.json();
    return all.filter((p) => !p.draft).sort((a, b) => (a.date < b.date ? 1 : -1));
  },
  date(iso) {
    const d = new Date(iso + "T00:00:00");
    return isNaN(d) ? iso : d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  },
  esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); },
};
