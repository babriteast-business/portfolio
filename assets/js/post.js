// Single post: post.html?p=<slug> loads posts/<slug>.md and renders the Markdown.
(async function () {
  const slug = new URLSearchParams(location.search).get("p") || "";
  const $ = (id) => document.getElementById(id);
  const fail = (msg) => { $("pTitle").textContent = "Post not found"; $("pBody").innerHTML = `<p>${msg} <a href="blog.html">Back to all posts</a></p>`; };
  if (!/^[a-z0-9-]+$/.test(slug)) return fail("That link doesn't point to a post.");
  let meta = {}, text;
  try {
    const idx = await Blog.index().catch(() => []);
    meta = idx.find((p) => p.slug === slug) || {};
    const r = await fetch(`posts/${slug}.md`, { cache: "no-cache" });
    if (!r.ok) return fail("This post may have been moved or renamed.");
    text = await r.text();
  } catch (e) { return fail("Couldn't load the post."); }
  // front matter
  const fm = text.match(/^﻿?---\s*\n([\s\S]*?)\n---\s*\n?/);
  if (fm) {
    fm[1].split("\n").forEach((l) => { const m = l.match(/^(\w+):\s*(.*)$/); if (m && meta[m[1]] === undefined) meta[m[1]] = m[2].trim(); });
    text = text.slice(fm[0].length);
  }
  const tags = Array.isArray(meta.tags) ? meta.tags : String(meta.tags || "").split(",").map((s) => s.trim()).filter(Boolean);
  const words = text.split(/\s+/).filter(Boolean).length;
  document.title = `${meta.title || "Post"}, Babrit Behera`;
  $("pTitle").textContent = meta.title || slug;
  $("pDate").textContent = meta.date ? Blog.date(meta.date) : "";
  $("pRead").textContent = `${Math.max(1, Math.round(words / 220))} min read`;
  $("pTags").innerHTML = tags.map((t) => `<a href="blog.html?tag=${encodeURIComponent(t)}"><span>${Blog.esc(t)}</span></a>`).join("");
  $("pBody").innerHTML = DOMPurify.sanitize(marked.parse(text));
  $("pBody").querySelectorAll("a[href^='http']").forEach((a) => { a.target = "_blank"; a.rel = "noopener"; });
  if (window.gsap && !window.REDUCED) gsap.from(".article > *", { opacity: 0, y: 14, duration: 0.8, stagger: 0.06, ease: "power3.out" });
})();
