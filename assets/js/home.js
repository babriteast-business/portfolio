// Home page: skill data, the 3D skill space, the access badge, the pipeline and latest posts.
import * as THREE from "../vendor/three.module.min.js";

const REDUCED = window.REDUCED;
const gsap = window.gsap;

// ---------- skills: one source for the grid and the 3D clusters ----------
// key: true = skills I lead with in interviews (shown brighter)
const SKILLS = [
  { name: "Generative AI & LLMs", short: "genai.llm", items: [
    ["Large language models", 1], ["Prompt engineering", 1], ["LoRA / QLoRA", 1], ["SFT", 1], ["DPO", 1],
    ["Hugging Face Transformers", 0], ["Unsloth", 0], ["LLM guardrails", 1], ["LLM evaluation & benchmarking", 1], ["BERT", 0], ["XLNet", 0]] },
  { name: "Agentic AI & automation", short: "agents", items: [
    ["Multi-agent systems", 1], ["Agent orchestration", 1], ["Agentic workflows", 1], ["Tool / function calling", 1],
    ["Claude & Claude Code", 1], ["OpenClaw", 0], ["n8n", 1], ["Workflow automation", 0], ["Conversational AI", 0], ["Rasa NLU", 0]] },
  { name: "LLM platforms & APIs", short: "llm.apis", items: [
    ["Anthropic Claude API", 1], ["OpenAI API", 1], ["Google Gemini API", 0], ["DeepSeek", 0], ["OpenRouter", 1]] },
  { name: "Machine learning & deep learning", short: "ml.dl", items: [
    ["PyTorch", 1], ["TensorFlow", 0], ["Keras", 0], ["Scikit-learn", 1], ["CNN", 0], ["RNN", 0], ["Transfer learning", 0],
    ["NLP", 1], ["Topic modelling", 0], ["Sentiment analysis", 0], ["Computer vision", 0], ["YOLO", 0], ["Tesseract OCR", 0],
    ["Predictive modelling", 0], ["AutoML", 0], ["A/B testing", 0]] },
  { name: "MLOps, DevSecOps & cloud", short: "mlops", items: [
    ["End-to-end ML lifecycle", 1], ["MLOps", 1], ["Model deployment", 1], ["CI/CD pipelines", 1], ["GitOps", 0], ["DevSecOps", 0],
    ["Docker & Compose", 1], ["Kubernetes", 0], ["Jenkins", 0], ["AWS", 1], ["Azure", 0], ["GCP", 0], ["Linux", 0]] },
  { name: "Programming, data & backend", short: "code.data", items: [
    ["Python", 1], ["SQL", 1], ["Java", 0], ["Pandas", 0], ["NumPy", 0], ["PySpark", 0], ["Apache Airflow", 0],
    ["FastAPI", 1], ["Flask", 0], ["REST APIs", 0], ["MongoDB", 0], ["Plotly & Dash", 0]] },
];

(function renderSkills() {
  const grid = document.getElementById("skillGrid");
  grid.innerHTML = SKILLS.map((g, i) => `
    <div class="skill reveal" data-cluster="${i}">
      <div class="skill-top"><h3>${g.name}</h3><span class="mono">${g.items.length}</span></div>
      <div class="chips">${g.items.map(([n, k]) => `<span class="chip${k ? " key" : ""}">${n}</span>`).join("")}</div>
    </div>`).join("");
})();

// ---------- 3D skill space ----------
let focusCluster = () => {};
(function skillSpace() {
  const canvas = document.getElementById("space");
  const labelsHost = document.getElementById("labels");
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
  } catch (e) { canvas.style.display = "none"; return; }
  renderer.setClearColor(0x0d0e10, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0d0e10, 9, 20);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0, 12);
  const world = new THREE.Group();   // moved left/right for layout
  const spin = new THREE.Group();    // rotated by drag + scroll
  world.add(spin); scene.add(world);

  const mobile = window.innerWidth < 700;
  const perSkill = mobile ? 9 : 16;

  // cluster centres on a tilted sphere, deterministic
  const rand = mulberry(7);
  const centres = SKILLS.map((_, i) => {
    const phi = Math.acos(1 - (2 * (i + 0.5)) / SKILLS.length);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    return new THREE.Vector3(Math.cos(theta) * Math.sin(phi), Math.cos(phi) * 0.85, Math.sin(theta) * Math.sin(phi)).multiplyScalar(2.35);
  });

  const pos = [], start = [], cluster = [], seed = [];
  SKILLS.forEach((g, ci) => {
    const n = g.items.length * perSkill;
    for (let k = 0; k < n; k++) {
      const r = Math.pow(rand(), 0.7) * 0.78;
      const u = rand() * 2 - 1, a = rand() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      const p = centres[ci].clone().add(new THREE.Vector3(s * Math.cos(a), u, s * Math.sin(a)).multiplyScalar(r));
      pos.push(p.x, p.y, p.z);
      start.push((rand() - 0.5) * 22, (rand() - 0.5) * 14, (rand() - 0.5) * 16);
      cluster.push(ci); seed.push(rand());
    }
  });

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("aStart", new THREE.Float32BufferAttribute(start, 3));
  geo.setAttribute("aCluster", new THREE.Float32BufferAttribute(cluster, 1));
  geo.setAttribute("aSeed", new THREE.Float32BufferAttribute(seed, 1));

  const uniforms = {
    uProgress: { value: REDUCED ? 1 : 0 },
    uFocus: { value: -1 },
    uFocusMix: { value: 0 },
    uPixel: { value: renderer.getPixelRatio() },
    uTime: { value: 0 },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false,
    vertexShader: `
      attribute vec3 aStart; attribute float aCluster; attribute float aSeed;
      uniform float uProgress, uFocus, uFocusMix, uPixel, uTime;
      varying float vHot; varying float vAlpha;
      void main(){
        float t = clamp((uProgress * 1.6) - aSeed * 0.6, 0.0, 1.0);
        t = 1.0 - pow(1.0 - t, 3.0);
        vec3 p = mix(aStart, position, t);
        p += 0.03 * vec3(sin(uTime*0.7 + aSeed*40.0), cos(uTime*0.6 + aSeed*30.0), sin(uTime*0.5 + aSeed*20.0));
        vHot = (abs(aCluster - uFocus) < 0.5) ? uFocusMix : 0.0;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        float size = mix(2.2, 3.4, vHot) + aSeed * 1.2;
        gl_PointSize = size * uPixel * (9.0 / -mv.z);
        vAlpha = mix(0.15, 1.0, t) * smoothstep(20.0, 8.0, -mv.z);
      }`,
    fragmentShader: `
      varying float vHot; varying float vAlpha;
      void main(){
        vec2 c = gl_PointCoord - 0.5; float d = length(c);
        if (d > 0.5) discard;
        float a = smoothstep(0.5, 0.15, d);
        vec3 base = vec3(0.60, 0.63, 0.67);
        vec3 hot = vec3(0.949, 0.663, 0.231);
        gl_FragColor = vec4(mix(base, hot, vHot), a * vAlpha * mix(0.55, 0.95, vHot));
      }`,
  });
  spin.add(new THREE.Points(geo, mat));

  // nearest-neighbour edges inside each cluster: the "attention" web
  const edge = [];
  const P = (i) => [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
  for (let i = 0; i < cluster.length; i += 3) {
    let best = [], a = P(i);
    for (let j = 0; j < cluster.length; j++) {
      if (j === i || cluster[j] !== cluster[i]) continue;
      const b = P(j), d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
      if (best.length < 2) best.push([d, j]); else if (d < best[1][0]) best[1] = [d, j];
      best.sort((x, y) => x[0] - y[0]);
    }
    best.forEach(([, j]) => edge.push(...a, ...P(j)));
  }
  // links between cluster centres
  for (let i = 0; i < centres.length; i++) for (let j = i + 1; j < centres.length; j++) {
    if (centres[i].distanceTo(centres[j]) < 3.4) edge.push(centres[i].x, centres[i].y, centres[i].z, centres[j].x, centres[j].y, centres[j].z);
  }
  const lgeo = new THREE.BufferGeometry();
  lgeo.setAttribute("position", new THREE.Float32BufferAttribute(edge, 3));
  const lmat = new THREE.LineBasicMaterial({ color: 0x9aa1aa, transparent: true, opacity: 0, depthWrite: false });
  spin.add(new THREE.LineSegments(lgeo, lmat));

  // HTML labels pinned to cluster centres
  const labels = SKILLS.map((g) => {
    const el = document.createElement("div");
    el.className = "cluster-label"; el.textContent = g.short; el.style.opacity = 0;
    labelsHost.appendChild(el); return el;
  });

  // layout
  let W = 0, H = 0;
  function resize() {
    W = canvas.clientWidth; H = canvas.clientHeight;
    renderer.setSize(W, H, false);
    camera.aspect = W / H; camera.updateProjectionMatrix();
    const wide = W > 1000;
    world.position.set(wide ? 3.1 : 0, wide ? 0 : (W < 700 ? 2.7 : 1.9), 0);
    world.scale.setScalar(wide ? 1 : W < 700 ? 0.68 : 0.9);
  }
  resize(); window.addEventListener("resize", resize);

  // drag to rotate, with inertia
  let rx = -0.25, ry = 0.6, vx = 0, vy = 0, dragging = false, lx = 0, ly = 0, scrollTilt = 0;
  const hero = document.querySelector(".hero");
  hero.addEventListener("pointerdown", (e) => {
    if (e.target.closest("a,button")) return;
    dragging = true; lx = e.clientX; ly = e.clientY; hero.setPointerCapture?.(e.pointerId);
  });
  hero.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    vy = (e.clientX - lx) * 0.005; vx = (e.clientY - ly) * 0.004; lx = e.clientX; ly = e.clientY;
    ry += vy; rx = Math.max(-1.1, Math.min(1.1, rx + vx));
  });
  const stop = () => (dragging = false);
  hero.addEventListener("pointerup", stop); hero.addEventListener("pointercancel", stop);

  // focus a cluster (amber) - cycles slowly, and follows the skills grid on hover
  let focus = 0, manual = false;
  focusCluster = (i, isManual) => {
    manual = !!isManual;
    if (i === focus && uniforms.uFocusMix.value > 0.9) return;
    if (!gsap || REDUCED) { focus = i; uniforms.uFocus.value = i; uniforms.uFocusMix.value = 1; return; }
    gsap.to(uniforms.uFocusMix, { value: 0, duration: 0.35, ease: "power2.out", onComplete: () => {
      focus = i; uniforms.uFocus.value = i;
      gsap.to(uniforms.uFocusMix, { value: 1, duration: 0.6, ease: "power2.out" });
    } });
  };
  document.querySelectorAll("[data-cluster]").forEach((el) => {
    el.addEventListener("mouseenter", () => focusCluster(+el.dataset.cluster, true));
    el.addEventListener("mouseleave", () => (manual = false));
  });

  // intro: points converge from noise into clusters, like a run converging
  const intro = () => {
    if (!gsap || REDUCED) { lmat.opacity = 0.07; labels.forEach((l) => (l.style.opacity = 1)); focusCluster(0); return; }
    const tl = gsap.timeline({ delay: 0.15 });
    tl.to(uniforms.uProgress, { value: 1, duration: 2.6, ease: "power2.inOut" })
      .to(lmat, { opacity: 0.07, duration: 1.2, ease: "power1.out" }, "-=1.0")
      .to(labels, { opacity: 1, duration: 0.6, stagger: 0.06, ease: "power2.out" }, "-=0.8")
      .add(() => focusCluster(0), "-=0.4")
      .from(".hero-text > *", { opacity: 0, y: 14, duration: 0.8, stagger: 0.07, ease: "power3.out" }, 0.2);
  };
  intro();
  if (gsap && !REDUCED) {
    setInterval(() => { if (!manual && visible) focusCluster((focus + 1) % SKILLS.length); }, 3400);
    gsap.registerPlugin(ScrollTrigger);
    ScrollTrigger.create({ trigger: hero, start: "top top", end: "bottom top", scrub: true,
      onUpdate: (s) => { scrollTilt = s.progress; } });
  } else if (!REDUCED) { focusCluster(0); }

  // render loop, paused when the hero is off screen
  let visible = true;
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(hero);
  const v = new THREE.Vector3(), clock = new THREE.Clock();
  function frame() {
    requestAnimationFrame(frame);
    if (!visible) return;
    const dt = Math.min(clock.getDelta(), 0.05);
    uniforms.uTime.value += dt;
    if (!dragging) { vy *= 0.94; vx *= 0.94; ry += vy + (REDUCED ? 0 : dt * 0.06); rx = Math.max(-1.1, Math.min(1.1, rx + vx)); }
    spin.rotation.set(rx + scrollTilt * 0.6, ry, 0);
    camera.position.z = 12 + scrollTilt * 4;
    renderer.render(scene, camera);
    // pin labels
    centres.forEach((c, i) => {
      v.copy(c).applyMatrix4(spin.matrixWorld).project(camera);
      const x = (v.x * 0.5 + 0.5) * W, y = (-v.y * 0.5 + 0.5) * H;
      labels[i].style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
      labels[i].classList.toggle("hot", i === focus && uniforms.uFocusMix.value > 0.5);
      labels[i].style.zIndex = v.z < 0.98 ? 1 : 0;
    });
  }
  frame();

  function mulberry(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
})();

// ---------- access badge: tilts toward the pointer ----------
(function badge() {
  const stage = document.getElementById("badgeStage"), card = document.getElementById("badge");
  const bars = document.getElementById("barcode");
  // barcode from the name, so it's always the same
  const code = [...("BABRIT BEHERA GEN AI")].map((c) => c.charCodeAt(0));
  bars.innerHTML = code.map((n) => `<i style="width:${1 + (n % 3)}px;opacity:${0.5 + (n % 5) / 10}"></i>`).join("");
  if (!gsap || REDUCED || matchMedia("(hover: none)").matches) return;
  const rx = gsap.quickTo(card, "rotationX", { duration: 0.6, ease: "power3.out" });
  const ry = gsap.quickTo(card, "rotationY", { duration: 0.6, ease: "power3.out" });
  stage.addEventListener("pointermove", (e) => {
    const r = card.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
    ry((px - 0.5) * 22); rx((0.5 - py) * 16);
    card.style.setProperty("--gx", `${px * 100}%`); card.style.setProperty("--gy", `${py * 100}%`);
  });
  stage.addEventListener("pointerleave", () => { rx(0); ry(0); });
  gsap.from(card, { rotationY: -28, rotationX: 10, y: 30, opacity: 0, duration: 1.2, ease: "power3.out",
    scrollTrigger: { trigger: stage, start: "top 80%" } });
})();

// ---------- facts count up ----------
(function facts() {
  document.querySelectorAll("[data-count]").forEach((el) => {
    const to = +el.dataset.count, suffix = el.textContent.replace(/[0-9]/g, "");
    if (!gsap || REDUCED) return;
    const o = { v: 0 };
    gsap.to(o, { v: to, duration: 1.4, ease: "power2.out", scrollTrigger: { trigger: el, start: "top 90%" },
      onUpdate: () => (el.textContent = Math.round(o.v) + suffix) });
  });
})();

// ---------- pipeline: a token travels base model -> API ----------
(function pipeline() {
  const pipe = document.getElementById("pipe"), token = document.getElementById("pipeToken");
  const steps = [...pipe.querySelectorAll(".pipe-step")];
  if (!gsap || REDUCED) { steps.forEach((s) => s.classList.add("lit")); token.style.display = "none"; return; }
  const place = () => {
    const pr = pipe.getBoundingClientRect();
    return steps.map((s) => { const r = s.getBoundingClientRect(); return r.top - pr.top + r.height / 2 - 3; });
  };
  let tl;
  const build = () => {
    tl?.kill(); steps.forEach((s) => s.classList.remove("lit"));
    const ys = place();
    gsap.set(token, { top: ys[0], opacity: 1 });
    tl = gsap.timeline({ repeat: -1, repeatDelay: 1.6, paused: true });
    ys.forEach((y, i) => {
      tl.to(token, { top: y, duration: i ? 0.55 : 0.01, ease: "power2.inOut" })
        .add(() => steps[i].classList.add("lit"))
        .to({}, { duration: 0.35 });
    });
    tl.to(token, { opacity: 0, duration: 0.3 }).add(() => steps.forEach((s) => s.classList.remove("lit")));
    ScrollTrigger.create({ trigger: pipe, start: "top 75%", end: "bottom top", onToggle: (s) => (s.isActive ? tl.play() : tl.pause()) });
  };
  build();
  window.addEventListener("resize", () => setTimeout(build, 200));
})();

// ---------- latest posts ----------
(async function latest() {
  const host = document.getElementById("latestPosts");
  try {
    const posts = (await Blog.index()).slice(0, 3);
    host.innerHTML = posts.length ? posts.map((p) => `
      <a class="card-post reveal" href="post.html?p=${encodeURIComponent(p.slug)}">
        <span class="date">${Blog.date(p.date)}${p.minutes ? ` &nbsp; ${p.minutes} min read` : ""}</span>
        <h3>${Blog.esc(p.title)}</h3><p>${Blog.esc(p.summary || "")}</p></a>`).join("")
      : `<p class="empty">No posts yet.</p>`;
  } catch (e) { host.innerHTML = `<p class="empty">Posts load when the site is served over http.</p>`; }
  window.revealAll();
})();

window.revealAll();
