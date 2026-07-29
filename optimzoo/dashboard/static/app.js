const state = {
  ws: null,
  running: false,
  bounds: null,
  landscape: null,
  history: { iteration: [], best: [], mean: [], worst: [] },
  lastPopulation: null,
  lastBestPos: null,
  startTime: null,
};

const el = (id) => document.getElementById(id);

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

function lerpColor(a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ar = (pa >> 16) & 255, ag = (pa >> 8) & 255, ab = pa & 255;
  const br = (pb >> 16) & 255, bg = (pb >> 8) & 255, bb = pb & 255;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const b_ = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${b_})`;
}

function sequentialColor(t) {
  t = Math.max(0, Math.min(1, t));
  const n = SEQUENTIAL_BLUE.length - 1;
  const idx = Math.min(n - 1, Math.floor(t * n));
  const localT = t * n - idx;
  return lerpColor(SEQUENTIAL_BLUE[idx], SEQUENTIAL_BLUE[idx + 1], localT);
}

async function loadOptions() {
  const [algos, problems] = await Promise.all([
    fetch("/api/algorithms").then((r) => r.json()),
    fetch("/api/problems").then((r) => r.json()),
  ]);
  const algoSelect = el("algorithm");
  algos.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    algoSelect.appendChild(opt);
  });
  algoSelect.value = "ParticleSwarmOptimization";

  const problemSelect = el("problem");
  problems.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.dataset.fixed2d = p.fixed_2d;
    opt.textContent = p.fixed_2d ? `${p.name} (2D)` : p.name;
    problemSelect.appendChild(opt);
  });
  problemSelect.value = "Ackley";

  problemSelect.addEventListener("change", syncDimensionField);
  syncDimensionField();
}

function syncDimensionField() {
  const problemSelect = el("problem");
  const opt = problemSelect.selectedOptions[0];
  const dimInput = el("dimension");
  if (opt && opt.dataset.fixed2d === "true") {
    dimInput.value = 2;
    dimInput.disabled = true;
  } else {
    dimInput.disabled = false;
  }
}

async function loadLandscape(problem, dimension) {
  const canvas = el("landscape-canvas");
  const dim = dimension || 2;
  if (dim !== 2) {
    state.landscape = null;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = cssVar("--text-muted");
    ctx.font = "13px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(
      `Landscape view only available for 2D problems (dimension=${dim})`,
      canvas.width / 2,
      canvas.height / 2
    );
    return;
  }
  const res = await fetch(`/api/landscape?problem=${encodeURIComponent(problem)}&dimension=2&resolution=120`);
  state.landscape = await res.json();
  drawLandscape();
}

function drawLandscape() {
  const canvas = el("landscape-canvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!state.landscape) return;
  const { x, y, z } = state.landscape;
  const nx = x.length, ny = y.length;

  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      const v = z[i][j];
      if (v < zmin) zmin = v;
      if (v > zmax) zmax = v;
    }
  }

  const imgData = ctx.createImageData(w, h);
  const x0 = x[0], x1 = x[nx - 1], y0 = y[0], y1 = y[ny - 1];

  for (let py = 0; py < h; py++) {
    const yv = y1 - ((py + 0.5) / h) * (y1 - y0);
    const iy = Math.min(ny - 1, Math.max(0, Math.round(((yv - y0) / (y1 - y0)) * (ny - 1))));
    for (let px = 0; px < w; px++) {
      const xv = x0 + ((px + 0.5) / w) * (x1 - x0);
      const ix = Math.min(nx - 1, Math.max(0, Math.round(((xv - x0) / (x1 - x0)) * (nx - 1))));
      const v = z[iy][ix];
      const t = zmax > zmin ? (v - zmin) / (zmax - zmin) : 0;
      const color = sequentialColor(t);
      const m = color.match(/\d+/g).map(Number);
      const idx = (py * w + px) * 4;
      imgData.data[idx] = m[0];
      imgData.data[idx + 1] = m[1];
      imgData.data[idx + 2] = m[2];
      imgData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);

  state.landscapeBoundsForDraw = { x0, x1, y0, y1, w, h };
  drawPopulationOverlay();
}

function toCanvasXY(px, py) {
  const b = state.landscapeBoundsForDraw;
  if (!b) return null;
  const cx = ((px - b.x0) / (b.x1 - b.x0)) * b.w;
  const cy = b.h - ((py - b.y0) / (b.y1 - b.y0)) * b.h;
  return [cx, cy];
}

function drawPopulationOverlay() {
  const canvas = el("landscape-canvas");
  const ctx = canvas.getContext("2d");
  if (!state.landscapeBoundsForDraw) return;

  // Redraw base raster is expensive per-frame; instead keep a cached base layer.
  if (!state.baseImage) return;
  ctx.putImageData(state.baseImage, 0, 0);

  if (state.lastPopulation && state.lastPopulation.positions) {
    ctx.fillStyle = cssVar("--series-best");
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.9;
    state.lastPopulation.positions.forEach((p) => {
      const xy = toCanvasXY(p[0], p[1]);
      if (!xy) return;
      ctx.beginPath();
      ctx.arc(xy[0], xy[1], 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }

  if (state.lastBestPos) {
    const xy = toCanvasXY(state.lastBestPos[0], state.lastBestPos[1]);
    if (xy) {
      ctx.fillStyle = cssVar("--status-critical");
      ctx.strokeStyle = cssVar("--surface-1");
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(xy[0], xy[1], 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}

function cacheBaseImage() {
  const canvas = el("landscape-canvas");
  const ctx = canvas.getContext("2d");
  state.baseImage = ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function drawConvergence() {
  const canvas = el("convergence-canvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const pad = { left: 56, right: 16, top: 16, bottom: 30 };
  ctx.clearRect(0, 0, w, h);

  const { iteration, best, mean, worst } = state.history;
  if (iteration.length < 2) return;

  const allVals = best.concat(mean, worst).filter((v) => Number.isFinite(v));
  let vmin = Math.min(...allVals), vmax = Math.max(...allVals);
  if (vmin === vmax) { vmin -= 1; vmax += 1; }
  const pad10 = (vmax - vmin) * 0.05;
  vmin -= pad10; vmax += pad10;

  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const xAt = (i) => pad.left + (i / (iteration.length - 1)) * plotW;
  const yAt = (v) => pad.top + plotH - ((v - vmin) / (vmax - vmin)) * plotH;

  // gridlines
  ctx.strokeStyle = cssVar("--gridline");
  ctx.lineWidth = 1;
  const gridLines = 5;
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= gridLines; i++) {
    const v = vmin + (i / gridLines) * (vmax - vmin);
    const yy = yAt(v);
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(w - pad.right, yy);
    ctx.stroke();
    ctx.fillText(v.toExponential(1), pad.left - 8, yy);
  }

  ctx.strokeStyle = cssVar("--baseline");
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + plotH);
  ctx.lineTo(w - pad.right, pad.top + plotH);
  ctx.stroke();

  function drawSeries(values, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = xAt(i), y = yAt(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawSeries(worst, cssVar("--series-worst"));
  drawSeries(mean, cssVar("--series-mean"));
  drawSeries(best, cssVar("--series-best"));

  state.convergenceLayout = { pad, plotW, plotH, vmin, vmax, w, h };
}

function setupConvergenceHover() {
  const canvas = el("convergence-canvas");
  const tooltip = el("convergence-tooltip");
  canvas.addEventListener("mousemove", (ev) => {
    const layout = state.convergenceLayout;
    const { iteration, best, mean, worst } = state.history;
    if (!layout || iteration.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const mx = (ev.clientX - rect.left) * scaleX;
    const frac = (mx - layout.pad.left) / layout.plotW;
    const idx = Math.round(frac * (iteration.length - 1));
    if (idx < 0 || idx >= iteration.length) {
      tooltip.style.display = "none";
      return;
    }
    tooltip.style.display = "block";
    tooltip.style.left = `${ev.clientX - rect.left + 12}px`;
    tooltip.style.top = `${ev.clientY - rect.top + 8}px`;
    tooltip.innerHTML =
      `iter ${iteration[idx]}<br>` +
      `best: ${best[idx].toExponential(3)}<br>` +
      `mean: ${mean[idx].toExponential(3)}<br>` +
      `worst: ${worst[idx].toExponential(3)}`;
  });
  canvas.addEventListener("mouseleave", () => (tooltip.style.display = "none"));
}

function resetHistory() {
  state.history = { iteration: [], best: [], mean: [], worst: [] };
  state.lastPopulation = null;
  state.lastBestPos = null;
}

function fmt(v, digits = 4) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  return Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? v.toExponential(digits - 1) : v.toFixed(digits);
}

function setStatus(text, cls) {
  const line = el("status-line");
  line.textContent = text;
  line.className = cls || "";
}

async function startRun() {
  const algorithm = el("algorithm").value;
  const problem = el("problem").value;
  const dimension = parseInt(el("dimension").value, 10);
  const population_size = parseInt(el("population").value, 10);
  const max_iterations = parseInt(el("iterations").value, 10);
  const seedRaw = el("seed").value;
  const seed = seedRaw === "" ? null : parseInt(seedRaw, 10);

  resetHistory();
  el("stat-status").textContent = "starting";
  setStatus("Starting run…");

  await loadLandscape(problem, dimension);
  cacheBaseImage();

  const res = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ algorithm, problem, dimension, population_size, max_iterations, seed }),
  });
  if (!res.ok) {
    const err = await res.json();
    setStatus(`Error: ${err.detail}`, "error");
    return;
  }
  const { run_id } = await res.json();
  connectWebSocket(run_id);
}

function connectWebSocket(runId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}`);
  state.ws = ws;
  state.running = true;
  updateRunButtons();
  el("conn-indicator").textContent = "connected";
  state.startTime = performance.now();

  ws.onmessage = (ev) => {
    const event = JSON.parse(ev.data);
    handleEvent(event, runId);
  };
  ws.onclose = () => {
    el("conn-indicator").textContent = "disconnected";
    state.running = false;
    updateRunButtons();
  };
}

function handleEvent(event) {
  switch (event.type) {
    case "start":
      state.bounds = event.bounds;
      el("stat-status").textContent = "running";
      setStatus(`Running ${event.algorithm} on ${event.problem} (dim=${event.dimension})…`);
      break;
    case "iteration": {
      const h = state.history;
      h.iteration.push(event.iteration);
      h.best.push(event.best_fitness);
      h.mean.push(event.mean_fitness ?? event.best_fitness);
      h.worst.push(event.worst_fitness ?? event.best_fitness);

      state.lastPopulation = event.population;
      state.lastBestPos = event.best_position;

      el("stat-iteration").textContent = event.iteration;
      el("stat-best").textContent = fmt(event.best_fitness);
      el("stat-mean").textContent = fmt(event.mean_fitness);
      el("stat-std").textContent = fmt(event.std_fitness);
      el("stat-evals").textContent = event.n_evaluations;
      el("stat-elapsed").textContent = `${event.elapsed_seconds.toFixed(2)}s`;
      el("stat-ips").textContent = fmt(event.iteration / Math.max(event.elapsed_seconds, 1e-6), 1);

      drawPopulationOverlay();
      drawConvergence();
      break;
    }
    case "finish":
      el("stat-status").textContent = "finished";
      setStatus(
        `Finished: best_fitness=${fmt(event.best_fitness)} after ${event.n_iterations} iterations (${event.elapsed_seconds.toFixed(2)}s)`,
        "done"
      );
      break;
    case "stopped":
      el("stat-status").textContent = "stopped";
      setStatus("Run stopped by user.", "error");
      break;
    case "error":
      setStatus(`Error: ${event.message}`, "error");
      break;
    case "closed":
      state.running = false;
      updateRunButtons();
      break;
    default:
      break;
  }
}

function updateRunButtons() {
  el("start-btn").style.display = state.running ? "none" : "block";
  el("stop-btn").style.display = state.running ? "block" : "none";
  el("start-btn").disabled = state.running;
}

async function stopRun() {
  if (!state.currentRunId && !state.ws) return;
  el("stop-btn").disabled = true;
  const match = state.ws?.url.match(/\/ws\/runs\/(.+)$/);
  const runId = match ? match[1] : null;
  if (runId) {
    await fetch(`/api/runs/${runId}/stop`, { method: "POST" });
  }
  el("stop-btn").disabled = false;
}

el("start-btn").addEventListener("click", startRun);
el("stop-btn").addEventListener("click", stopRun);

setupConvergenceHover();
loadOptions();
