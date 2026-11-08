// Blog list: reads posts/index.json, newest first, filter by tag.
(async function () {
  const list = document.getElementById("postList"), filters = document.getElementById("filters");
  let posts = [];
  try { posts = await Blog.index(); }
  catch (e) { list.innerHTML = `<p class="empty">Couldn't load the posts. Open the site through its web address, not as a file.</p>`; return; }
  const tags = [...new Set(posts.flatMap((p) => p.tags || []))].sort();
  let active = new URLSearchParams(location.search).get("tag") || "All";
  const draw = () => {
    filters.innerHTML = ["All", ...tags].map((t) => `<button class="filter${t === active ? " on" : ""}" data-t="${Blog.esc(t)}">${Blog.esc(t)}</button>`).join("");
    const shown = active === "All" ? posts : posts.filter((p) => (p.tags || []).includes(active));
    list.innerHTML = shown.length ? shown.map((p) => `
      <a class="post-item reveal" href="post.html?p=${encodeURIComponent(p.slug)}">
        <span class="date">${Blog.date(p.date)}</span>
        <div><h2>${Blog.esc(p.title)}</h2><p>${Blog.esc(p.summary || "")}</p>
          <div class="tags">${(p.tags || []).map((t) => `<span>${Blog.esc(t)}</span>`).join("")}</div></div>
        <span class="read">${p.minutes ? p.minutes + " min read" : ""}</span>
      </a>`).join("") : `<p class="empty">No posts with that tag yet.</p>`;
    window.revealAll(list);
  };
  filters.addEventListener("click", (e) => {
    const b = e.target.closest("[data-t]"); if (!b) return;
    active = b.dataset.t; history.replaceState(null, "", active === "All" ? "blog.html" : `blog.html?tag=${encodeURIComponent(active)}`); draw();
  });
  draw();
  if (window.gsap && !window.REDUCED) gsap.from(".page-head > *", { opacity: 0, y: 14, duration: 0.8, stagger: 0.07, ease: "power3.out" });
})();
