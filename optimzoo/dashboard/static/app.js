// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

function newRunState() {
  return {
    runId: null,
    meta: null, // {algorithm, problem, dimension, bounds, ...}
    events: [], // ordered "iteration" events only (start/finish/etc handled separately)
    finished: false,
    finishEvent: null,
    landscape: null,
    landscapeBase: { x0: 0, x1: 1, y0: 0, y1: 1, w: 0, h: 0 },
    baseImage: null,
    view: { scale: 1, panX: 0, panY: 0 }, // landscape zoom/pan (in canvas pixels, applied on top of base fit)
    convergenceZoom: null, // {i0, i1} index range into events, or null = full range
  };
}

const state = {
  ws: null,
  running: false,
  scrubIndex: -1, // -1 = "live" (follow latest); else a fixed index into runA.events
  playTimer: null,
  compareMode: false,
  runA: newRunState(),
  runB: newRunState(),
  historyCache: [],
};

const el = (id) => document.getElementById(id);

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---------------------------------------------------------------------------
// Color ramp (sequential blue, from the validated palette)
// ---------------------------------------------------------------------------

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

function fmt(v, digits = 4) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  return Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? v.toExponential(digits - 1) : v.toFixed(digits);
}

// ---------------------------------------------------------------------------
// Options loading
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Landscape (fitness heatmap + population overlay), with zoom/pan
// ---------------------------------------------------------------------------

async function loadLandscape(run, problem, dimension) {
  const dim = dimension || 2;
  if (dim !== 2) {
    run.landscape = null;
    return;
  }
  const res = await fetch(`/api/landscape?problem=${encodeURIComponent(problem)}&dimension=2&resolution=120`);
  run.landscape = await res.json();
}

function rasterizeLandscape(run, canvas) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!run.landscape) return;

  const { x, y, z } = run.landscape;
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
  run.baseImage = ctx.getImageData(0, 0, w, h);
  run.landscapeBase = { x0, x1, y0, y1, w, h };
}

// Map data-space (x,y) to canvas pixel, honoring the current zoom/pan view.
function dataToCanvas(run, px, py) {
  const b = run.landscapeBase;
  if (!b.w) return null;
  const baseX = ((px - b.x0) / (b.x1 - b.x0)) * b.w;
  const baseY = b.h - ((py - b.y0) / (b.y1 - b.y0)) * b.h;
  return [baseX * run.view.scale + run.view.panX, baseY * run.view.scale + run.view.panY];
}

function drawLandscapeFrame(run, canvas, eventIndex) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!run.baseImage) return;

  ctx.save();
  ctx.translate(run.view.panX, run.view.panY);
  ctx.scale(run.view.scale, run.view.scale);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(canvas._baseCanvas, 0, 0);
  ctx.restore();

  const event = eventIndex >= 0 && eventIndex < run.events.length ? run.events[eventIndex] : null;
  if (!event) return;

  if (event.population && event.population.positions) {
    // Orange (not the landscape's own blue ramp) so population dots stay
    // visible regardless of how light/dark the heatmap is at that point;
    // a white halo plus dark outline keeps them legible in both cases.
    ctx.fillStyle = cssVar("--series-mean");
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    event.population.positions.forEach((p) => {
      const xy = dataToCanvas(run, p[0], p[1]);
      if (!xy) return;
      ctx.beginPath();
      ctx.arc(xy[0], xy[1], 3.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }

  if (event.best_position) {
    const xy = dataToCanvas(run, event.best_position[0], event.best_position[1]);
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

// Keep an offscreen canvas holding the raw raster so zoom/pan can redraw cheaply.
function ensureBaseCanvas(canvas) {
  if (!canvas._baseCanvas) {
    canvas._baseCanvas = document.createElement("canvas");
    canvas._baseCanvas.width = canvas.width;
    canvas._baseCanvas.height = canvas.height;
  }
  return canvas._baseCanvas;
}

function rebuildLandscape(run, canvas) {
  const base = ensureBaseCanvas(canvas);
  rasterizeLandscape(run, base);
}

function setupLandscapeInteraction(canvas, getRun, onChange) {
  let dragging = false;
  let lastX = 0, lastY = 0;

  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const run = getRun();
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (ev.clientX - rect.left) * scaleX;
    const my = (ev.clientY - rect.top) * scaleY;

    const zoomFactor = ev.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.min(20, Math.max(1, run.view.scale * zoomFactor));

    // zoom around the cursor position
    const dataX = (mx - run.view.panX) / run.view.scale;
    const dataY = (my - run.view.panY) / run.view.scale;
    run.view.scale = newScale;
    run.view.panX = mx - dataX * newScale;
    run.view.panY = my - dataY * newScale;
    clampPan(run, canvas);
    onChange();
  }, { passive: false });

  canvas.addEventListener("mousedown", (ev) => {
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
    canvas.classList.add("dragging");
  });
  window.addEventListener("mousemove", (ev) => {
    if (!dragging) return;
    const run = getRun();
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    run.view.panX += (ev.clientX - lastX) * scaleX;
    run.view.panY += (ev.clientY - lastY) * scaleY;
    lastX = ev.clientX;
    lastY = ev.clientY;
    clampPan(run, canvas);
    onChange();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("dblclick", () => {
    const run = getRun();
    run.view = { scale: 1, panX: 0, panY: 0 };
    onChange();
  });
}

function clampPan(run, canvas) {
  const w = canvas.width, h = canvas.height;
  const scaledW = w * run.view.scale;
  const scaledH = h * run.view.scale;
  const minPanX = Math.min(0, w - scaledW);
  const minPanY = Math.min(0, h - scaledH);
  run.view.panX = Math.min(0, Math.max(minPanX, run.view.panX));
  run.view.panY = Math.min(0, Math.max(minPanY, run.view.panY));
}

function setupLandscapeTooltip(canvas, getRun, tooltipEl) {
  canvas.addEventListener("mousemove", (ev) => {
    const run = getRun();
    const idx = currentEventIndex(run);
    const event = idx >= 0 && idx < run.events.length ? run.events[idx] : null;
    if (!event || !event.population) {
      tooltipEl.style.display = "none";
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (ev.clientX - rect.left) * scaleX;
    const my = (ev.clientY - rect.top) * scaleY;

    // find nearest population point in canvas space
    let nearest = null, nearestDist = 14; // px threshold
    event.population.positions.forEach((p, i) => {
      const xy = dataToCanvas(run, p[0], p[1]);
      if (!xy) return;
      const d = Math.hypot(xy[0] - mx, xy[1] - my);
      if (d < nearestDist) {
        nearestDist = d;
        nearest = { p, fitness: event.population.fitness ? event.population.fitness[i] : null };
      }
    });

    if (!nearest) {
      tooltipEl.style.display = "none";
      return;
    }
    tooltipEl.style.display = "block";
    tooltipEl.style.left = `${ev.clientX - rect.left + 12}px`;
    tooltipEl.style.top = `${ev.clientY - rect.top + 8}px`;
    tooltipEl.innerHTML =
      `x: ${nearest.p[0].toFixed(3)}<br>y: ${nearest.p[1].toFixed(3)}` +
      (nearest.fitness !== null ? `<br>fitness: ${fmt(nearest.fitness)}` : "");
  });
  canvas.addEventListener("mouseleave", () => (tooltipEl.style.display = "none"));
}

// ---------------------------------------------------------------------------
// Convergence chart, with drag-to-zoom on the x (iteration) axis
// ---------------------------------------------------------------------------

function computeConvergenceSeries(run) {
  const iteration = [], best = [], mean = [], worst = [];
  run.events.forEach((e) => {
    iteration.push(e.iteration);
    best.push(e.best_fitness);
    mean.push(e.mean_fitness ?? e.best_fitness);
    worst.push(e.worst_fitness ?? e.best_fitness);
  });
  return { iteration, best, mean, worst };
}

function drawConvergence(run, canvas, upToIndex) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const pad = { left: 56, right: 16, top: 16, bottom: 30 };
  ctx.clearRect(0, 0, w, h);

  const series = computeConvergenceSeries(run);
  const n = upToIndex >= 0 ? upToIndex + 1 : series.iteration.length;
  if (n < 2) {
    canvas._layout = null;
    return;
  }

  let i0 = 0, i1 = n - 1;
  if (run.convergenceZoom) {
    i0 = Math.max(0, run.convergenceZoom.i0);
    i1 = Math.min(n - 1, run.convergenceZoom.i1);
    if (i1 - i0 < 1) i1 = Math.min(n - 1, i0 + 1);
  }

  const iteration = series.iteration.slice(i0, i1 + 1);
  const best = series.best.slice(i0, i1 + 1);
  const mean = series.mean.slice(i0, i1 + 1);
  const worst = series.worst.slice(i0, i1 + 1);

  const allVals = best.concat(mean, worst).filter((v) => Number.isFinite(v));
  let vmin = Math.min(...allVals), vmax = Math.max(...allVals);
  if (vmin === vmax) { vmin -= 1; vmax += 1; }
  const pad10 = (vmax - vmin) * 0.05;
  vmin -= pad10; vmax += pad10;

  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const xAt = (localI) => pad.left + (localI / Math.max(1, iteration.length - 1)) * plotW;
  const yAt = (v) => pad.top + plotH - ((v - vmin) / (vmax - vmin)) * plotH;

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

  // marker for the currently scrubbed iteration, if within this zoomed range
  if (upToIndex >= i0 && upToIndex <= i1) {
    const localI = upToIndex - i0;
    const x = xAt(localI);
    ctx.strokeStyle = cssVar("--text-secondary");
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  canvas._layout = { pad, plotW, plotH, vmin, vmax, w, h, i0, i1, iteration, best, mean, worst };
}

function setupConvergenceInteraction(canvas, getRun, tooltipEl, onZoomChange) {
  let dragStart = null;
  let dragCurrent = null;

  function indexFromClientX(clientX) {
    const layout = canvas._layout;
    if (!layout) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const mx = (clientX - rect.left) * scaleX;
    const frac = (mx - layout.pad.left) / layout.plotW;
    const localIdx = Math.round(frac * (layout.iteration.length - 1));
    return layout.i0 + Math.max(0, Math.min(layout.iteration.length - 1, localIdx));
  }

  canvas.addEventListener("mousedown", (ev) => {
    dragStart = ev.clientX;
    dragCurrent = ev.clientX;
  });

  window.addEventListener("mousemove", (ev) => {
    if (dragStart !== null) {
      dragCurrent = ev.clientX;
      redrawSelectionOverlay();
      return;
    }

    // Only the canvas's own hover area drives the tooltip — a window-level
    // listener is used purely so dragging can continue past the canvas edge;
    // without this bounds check the tooltip would show stale content whenever
    // the pointer crosses over any other part of the page.
    const rect = canvas.getBoundingClientRect();
    const withinBounds =
      ev.clientX >= rect.left && ev.clientX <= rect.right && ev.clientY >= rect.top && ev.clientY <= rect.bottom;
    if (!withinBounds) {
      tooltipEl.style.display = "none";
      return;
    }

    const layout = canvas._layout;
    if (!layout) {
      tooltipEl.style.display = "none";
      return;
    }
    const idx = indexFromClientX(ev.clientX);
    if (idx === null) {
      tooltipEl.style.display = "none";
      return;
    }
    const run = getRun();
    const e = run.events[idx];
    if (!e) {
      tooltipEl.style.display = "none";
      return;
    }
    tooltipEl.style.display = "block";
    tooltipEl.style.left = `${ev.clientX - rect.left + 12}px`;
    tooltipEl.style.top = `${ev.clientY - rect.top + 8}px`;
    tooltipEl.innerHTML =
      `iter ${e.iteration}<br>` +
      `best: ${fmt(e.best_fitness, 5)}<br>` +
      `mean: ${fmt(e.mean_fitness, 5)}<br>` +
      `worst: ${fmt(e.worst_fitness, 5)}`;
  });

  function redrawSelectionOverlay() {
    const layout = canvas._layout;
    if (!layout || dragStart === null) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const x0 = (Math.min(dragStart, dragCurrent) - rect.left) * scaleX;
    const x1 = (Math.max(dragStart, dragCurrent) - rect.left) * scaleX;

    const run = getRun();
    const upToIndex = currentEventIndex(run);
    drawConvergence(run, canvas, upToIndex);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "rgba(42, 120, 214, 0.15)";
    ctx.fillRect(x0, layout.pad.top, x1 - x0, layout.plotH);
  }

  window.addEventListener("mouseup", () => {
    if (dragStart === null) return;
    const movedEnough = Math.abs(dragCurrent - dragStart) > 6;
    if (movedEnough) {
      const i0 = indexFromClientX(Math.min(dragStart, dragCurrent));
      const i1 = indexFromClientX(Math.max(dragStart, dragCurrent));
      if (i0 !== null && i1 !== null && i1 > i0) {
        const run = getRun();
        run.convergenceZoom = { i0, i1 };
        onZoomChange();
      }
    }
    dragStart = null;
    dragCurrent = null;
  });

  canvas.addEventListener("mouseleave", () => {
    if (dragStart === null) tooltipEl.style.display = "none";
  });

  canvas.addEventListener("dblclick", () => {
    const run = getRun();
    run.convergenceZoom = null;
    onZoomChange();
  });
}

// ---------------------------------------------------------------------------
// Scrubber
// ---------------------------------------------------------------------------

function currentEventIndex(run) {
  if (state.scrubIndex >= 0) return Math.min(state.scrubIndex, run.events.length - 1);
  return run.events.length - 1;
}

function syncScrubberRange() {
  const run = state.runA;
  const slider = el("scrub-slider");
  const maxIdx = Math.max(0, run.events.length - 1);
  slider.max = String(maxIdx);
  if (state.scrubIndex === -1 || state.scrubIndex > maxIdx) {
    slider.value = String(maxIdx);
  } else {
    slider.value = String(state.scrubIndex);
  }
  updateScrubLabel();
}

function updateScrubLabel() {
  const run = state.runA;
  const idx = currentEventIndex(run);
  const e = run.events[idx];
  const label = el("scrub-label");
  if (!e) {
    label.textContent = "iteration –";
  } else {
    const liveTag = state.scrubIndex === -1 && state.running ? " (live)" : "";
    label.textContent = `iteration ${e.iteration}${liveTag}`;
  }
}

function renderAll() {
  const run = state.runA;
  const idx = currentEventIndex(run);
  drawLandscapeFrame(run, el("landscape-canvas"), idx);
  drawConvergence(run, el("convergence-canvas"), idx);
  updateStatsPanel(run, idx);
  syncScrubberRange();

  if (state.compareMode) {
    drawCompareChart();
  }
}

function updateStatsPanel(run, idx) {
  const e = run.events[idx];
  if (!e) return;
  el("stat-iteration").textContent = e.iteration;
  el("stat-best").textContent = fmt(e.best_fitness);
  el("stat-mean").textContent = fmt(e.mean_fitness);
  el("stat-std").textContent = fmt(e.std_fitness);
  el("stat-evals").textContent = e.n_evaluations;
  el("stat-elapsed").textContent = `${e.elapsed_seconds.toFixed(2)}s`;
  el("stat-ips").textContent = fmt(e.iteration / Math.max(e.elapsed_seconds, 1e-6), 1);
}

function setupScrubberControls() {
  const slider = el("scrub-slider");
  slider.addEventListener("input", () => {
    stopPlayback();
    const maxIdx = state.runA.events.length - 1;
    const v = parseInt(slider.value, 10);
    state.scrubIndex = v >= maxIdx ? -1 : v;
    renderAll();
  });

  el("scrub-prev").addEventListener("click", () => {
    stopPlayback();
    const idx = currentEventIndex(state.runA);
    state.scrubIndex = Math.max(0, idx - 1);
    renderAll();
  });

  el("scrub-next").addEventListener("click", () => {
    stopPlayback();
    const maxIdx = state.runA.events.length - 1;
    const idx = currentEventIndex(state.runA);
    const next = idx + 1;
    state.scrubIndex = next >= maxIdx ? -1 : next;
    renderAll();
  });

  el("scrub-play").addEventListener("click", () => {
    if (state.playTimer) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });

  window.addEventListener("keydown", (ev) => {
    if (document.activeElement && ["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
    if (ev.key === "ArrowLeft") {
      stopPlayback();
      const idx = currentEventIndex(state.runA);
      state.scrubIndex = Math.max(0, idx - 1);
      renderAll();
    } else if (ev.key === "ArrowRight") {
      stopPlayback();
      const maxIdx = state.runA.events.length - 1;
      const idx = currentEventIndex(state.runA);
      const next = idx + 1;
      state.scrubIndex = next >= maxIdx ? -1 : next;
      renderAll();
    } else if (ev.key === " ") {
      ev.preventDefault();
      if (state.playTimer) stopPlayback();
      else startPlayback();
    }
  });
}

function startPlayback() {
  const maxIdx = state.runA.events.length - 1;
  if (maxIdx < 1) return;
  if (currentEventIndex(state.runA) >= maxIdx) state.scrubIndex = 0;
  el("scrub-play").innerHTML = "&#10074;&#10074;";
  state.playTimer = setInterval(() => {
    const max = state.runA.events.length - 1;
    const idx = currentEventIndex(state.runA);
    const next = idx + 1;
    if (next >= max) {
      state.scrubIndex = -1;
      stopPlayback();
    } else {
      state.scrubIndex = next;
    }
    renderAll();
  }, 90);
}

function stopPlayback() {
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
    el("scrub-play").innerHTML = "&#9654;";
  }
}

// ---------------------------------------------------------------------------
// Run lifecycle: start / stream / stop
// ---------------------------------------------------------------------------

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

  state.runA = newRunState();
  state.scrubIndex = -1;
  el("stat-status").textContent = "starting";
  setStatus("Starting run…");

  await loadLandscape(state.runA, problem, dimension);
  rebuildLandscape(state.runA, el("landscape-canvas"));

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
  state.runA.runId = run_id;
  el("landscape-run-label").textContent = `#${run_id}`;
  el("convergence-run-label").textContent = `#${run_id}`;
  connectWebSocket(run_id);
}

function connectWebSocket(runId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}`);
  state.ws = ws;
  state.running = true;
  updateRunButtons();
  el("conn-indicator").textContent = "connected";

  ws.onmessage = (ev) => {
    const event = JSON.parse(ev.data);
    handleEvent(event);
  };
  ws.onclose = () => {
    el("conn-indicator").textContent = "disconnected";
    state.running = false;
    updateRunButtons();
    refreshHistory();
  };
}

function handleEvent(event) {
  const run = state.runA;
  switch (event.type) {
    case "start":
      run.meta = event;
      el("stat-status").textContent = "running";
      setStatus(`Running ${event.algorithm} on ${event.problem} (dim=${event.dimension})…`);
      break;
    case "iteration":
      run.events.push(event);
      if (state.scrubIndex === -1) {
        renderAll();
      } else {
        // still update stats/scrubber range even while looking at the past
        syncScrubberRange();
      }
      break;
    case "finish":
      run.finished = true;
      run.finishEvent = event;
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
  if (!state.runA.runId) return;
  el("stop-btn").disabled = true;
  await fetch(`/api/runs/${state.runA.runId}/stop`, { method: "POST" });
  el("stop-btn").disabled = false;
}

// ---------------------------------------------------------------------------
// Run history sidebar
// ---------------------------------------------------------------------------

async function refreshHistory() {
  const runs = await fetch("/api/runs").then((r) => r.json());
  state.historyCache = runs;
  renderHistoryList();
}

function renderHistoryList() {
  const container = el("run-list");
  container.innerHTML = "";
  if (state.historyCache.length === 0) {
    container.innerHTML = '<div class="empty-hint">No runs yet this session.</div>';
    return;
  }
  state.historyCache.forEach((r) => {
    const item = document.createElement("div");
    item.className = "run-item";
    if (state.runA.runId === r.run_id) item.classList.add("selected");
    if (state.runB.runId === r.run_id) item.classList.add("selected-b");

    const statusBadge = r.done ? "finished" : "running";
    item.innerHTML = `
      <div class="title">${r.algorithm}</div>
      <div class="meta">${r.problem} · dim ${r.dimension} · pop ${r.population_size}</div>
      <div class="badges">
        <span class="run-badge">${statusBadge}</span>
        <span class="run-badge">best ${fmt(r.best_fitness, 3)}</span>
        <span class="run-badge">${r.n_iterations_completed}/${r.max_iterations} it</span>
      </div>
    `;
    item.addEventListener("click", () => loadRunIntoSlot(r.run_id, state.compareMode ? "B" : "A"));
    container.appendChild(item);
  });
}

// Reflect a loaded/replayed run's configuration in the run-config form, so it
// doesn't keep showing whatever was last typed in for a different run.
function syncConfigFormToRunSummary(summary) {
  el("algorithm").value = summary.algorithm;
  el("problem").value = summary.problem;
  syncDimensionField();
  el("dimension").value = summary.dimension;
  el("population").value = summary.population_size;
  el("iterations").value = summary.max_iterations;
  el("seed").value = summary.seed === null || summary.seed === undefined ? "" : summary.seed;
}

async function loadRunIntoSlot(runId, slot) {
  const full = await fetch(`/api/runs/${runId}`).then((r) => r.json());
  const runState = newRunState();
  runState.runId = runId;
  runState.meta = full.events.find((e) => e.type === "start") || null;
  runState.events = full.events.filter((e) => e.type === "iteration");
  runState.finished = full.done;
  runState.finishEvent = full.events.find((e) => e.type === "finish") || null;

  if (slot === "A") {
    state.runA = runState;
    state.scrubIndex = -1;
    if (runState.meta) {
      await loadLandscape(state.runA, runState.meta.problem, runState.meta.dimension);
      rebuildLandscape(state.runA, el("landscape-canvas"));
    }
    syncConfigFormToRunSummary(full);
    el("landscape-run-label").textContent = `#${runId}`;
    el("convergence-run-label").textContent = `#${runId}`;
    setStatus(`Viewing past run #${runId} (read-only replay).`, "done");
    el("stat-status").textContent = full.done ? "finished (replay)" : "running (replay)";
    renderAll();
  } else {
    state.runB = runState;
    if (runState.meta) {
      await loadLandscape(state.runB, runState.meta.problem, runState.meta.dimension);
    }
    drawCompareChart();
  }
  renderHistoryList();
}

// ---------------------------------------------------------------------------
// Compare mode
// ---------------------------------------------------------------------------

function drawCompareChart() {
  const canvas = el("compare-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const pad = { left: 60, right: 16, top: 16, bottom: 30 };
  ctx.clearRect(0, 0, w, h);

  const seriesA = computeConvergenceSeries(state.runA);
  const seriesB = state.runB.runId ? computeConvergenceSeries(state.runB) : null;
  if (seriesA.iteration.length < 2 && (!seriesB || seriesB.iteration.length < 2)) return;

  const allVals = [...seriesA.best];
  if (seriesB) allVals.push(...seriesB.best);
  const finiteVals = allVals.filter((v) => Number.isFinite(v));
  if (finiteVals.length === 0) return;
  let vmin = Math.min(...finiteVals), vmax = Math.max(...finiteVals);
  if (vmin === vmax) { vmin -= 1; vmax += 1; }
  const pad10 = (vmax - vmin) * 0.05;
  vmin -= pad10; vmax += pad10;

  const maxLen = Math.max(seriesA.iteration.length, seriesB ? seriesB.iteration.length : 0);
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const xAt = (i) => pad.left + (i / Math.max(1, maxLen - 1)) * plotW;
  const yAt = (v) => pad.top + plotH - ((v - vmin) / (vmax - vmin)) * plotH;

  ctx.strokeStyle = cssVar("--gridline");
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 5; i++) {
    const v = vmin + (i / 5) * (vmax - vmin);
    const yy = yAt(v);
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(w - pad.right, yy);
    ctx.stroke();
    ctx.fillText(v.toExponential(1), pad.left - 8, yy);
  }

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

  drawSeries(seriesA.best, cssVar("--series-best"));
  if (seriesB) drawSeries(seriesB.best, cssVar("--series-run-b"));

  canvas._layout = { pad, plotW, plotH, vmin, vmax, w, h, maxLen, seriesA, seriesB };
}

function setupCompareTooltip() {
  const canvas = el("compare-canvas");
  const tooltip = el("compare-tooltip");
  if (!canvas) return;
  canvas.addEventListener("mousemove", (ev) => {
    const layout = canvas._layout;
    if (!layout) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const mx = (ev.clientX - rect.left) * scaleX;
    const frac = (mx - layout.pad.left) / layout.plotW;
    const idx = Math.round(frac * (layout.maxLen - 1));
    if (idx < 0 || idx >= layout.maxLen) {
      tooltip.style.display = "none";
      return;
    }
    const aVal = layout.seriesA.best[idx];
    const bVal = layout.seriesB ? layout.seriesB.best[idx] : undefined;
    tooltip.style.display = "block";
    tooltip.style.left = `${ev.clientX - rect.left + 12}px`;
    tooltip.style.top = `${ev.clientY - rect.top + 8}px`;
    tooltip.innerHTML =
      `iter ${idx}<br>run A: ${aVal !== undefined ? fmt(aVal, 5) : "–"}` +
      (bVal !== undefined ? `<br>run B: ${fmt(bVal, 5)}` : "");
  });
  canvas.addEventListener("mouseleave", () => (tooltip.style.display = "none"));
}

function setupCompareToggle() {
  el("compare-checkbox").addEventListener("change", (ev) => {
    state.compareMode = ev.target.checked;
    el("compare-section").classList.toggle("active", state.compareMode);
    if (!state.compareMode) {
      state.runB = newRunState();
    }
    renderHistoryList();
    drawCompareChart();
  });
}

// ---------------------------------------------------------------------------
// Wire-up
// ---------------------------------------------------------------------------

el("start-btn").addEventListener("click", startRun);
el("stop-btn").addEventListener("click", stopRun);
el("refresh-history").addEventListener("click", refreshHistory);
el("landscape-reset").addEventListener("click", () => {
  state.runA.view = { scale: 1, panX: 0, panY: 0 };
  renderAll();
});
el("convergence-reset").addEventListener("click", () => {
  state.runA.convergenceZoom = null;
  renderAll();
});

setupLandscapeInteraction(el("landscape-canvas"), () => state.runA, renderAll);
setupLandscapeTooltip(el("landscape-canvas"), () => state.runA, el("landscape-tooltip"));
setupConvergenceInteraction(el("convergence-canvas"), () => state.runA, el("convergence-tooltip"), renderAll);
setupScrubberControls();
setupCompareToggle();
setupCompareTooltip();

loadOptions();
refreshHistory();
setInterval(() => {
  if (!state.playTimer) refreshHistory();
}, 5000);
