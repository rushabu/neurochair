import { createScene } from './scene.js';
import { MicListener } from './audio.js';

const KEYWORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'];
const COMMANDS = {
  go:    { label: 'Drive forward', ico: '▲', c: '#4dff9a' },
  stop:  { label: 'Brake',         ico: '■', c: '#ff4d6a' },
  left:  { label: 'Turn left',     ico: '↰', c: '#3ef2ff' },
  right: { label: 'Turn right',    ico: '↱', c: '#3ef2ff' },
  up:    { label: 'Speed up',      ico: '+', c: '#8a6bff' },
  down:  { label: 'Slow down',     ico: '−', c: '#8a6bff' },
  on:    { label: 'Lights on',     ico: '☀', c: '#ffb547' },
  off:   { label: 'Lights off',    ico: '☾', c: '#ffb547' },
};
const WORD_COLORS = { ...Object.fromEntries(Object.entries(COMMANDS).map(([k, v]) => [k, v.c])), yes: '#8a93b8', no: '#8a93b8' };
const STOP_BIAS = 0.6; // "stop" only needs 60% of the normal confidence gate

const $ = (id) => document.getElementById(id);
const ui = {
  conf: 0.45, speedWords: true,
};

// ---------- toast ----------
let toastTimer;
function toast(msg, kind = '') {
  const t = $('toast');
  t.textContent = msg; t.className = `toast show ${kind}`;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = 'toast'), 2200);
}

// ---------- scene ----------
const world = createScene($('world'), {
  onBlocked: (what) => {
    toast(`⚠ ${what === 'boundary' ? 'Edge of area' : 'Obstacle ahead'} — auto-brake engaged`, 'danger');
    $('tmState').className = 'blocked';
    addLog('auto-brake', '#ff4d6a', false);
  },
});

// ---------- command execution ----------
function activeWords() {
  const words = ['go', 'stop', 'left', 'right', 'on', 'off'];
  if (ui.speedWords) words.push('up', 'down');
  return words;
}

function execute(word, source = 'voice') {
  const actions = {
    go: () => world.go(),
    stop: () => world.stop(),
    left: () => world.turn('left'),
    right: () => world.turn('right'),
    up: () => world.speed(+1),
    down: () => world.speed(-1),
    on: () => world.setLights(true),
    off: () => world.setLights(false),
  };
  if (!actions[word]) return;
  actions[word]();
  world.burst(COMMANDS[word].c, source === 'voice' ? 90 : 40);
  addLog(word, COMMANDS[word].c, source !== 'voice');
  const chip = document.querySelector(`.chip[data-w="${word}"]`);
  if (chip) { chip.classList.add('flash'); setTimeout(() => chip.classList.remove('flash'), 600); }
  if (word === 'stop') $('tmState').className = '';
}

function addLog(text, color, keyboard) {
  const log = $('log');
  const el = document.createElement('span');
  el.className = `log-item${keyboard ? ' kb' : ''}`;
  el.style.setProperty('--c', color);
  el.textContent = text;
  log.prepend(el);
  while (log.children.length > 6) log.lastChild.remove();
}

// ---------- chips ----------
function renderChips() {
  const wrap = $('chips');
  wrap.innerHTML = '';
  const active = activeWords();
  for (const [w, cmd] of Object.entries(COMMANDS)) {
    const b = document.createElement('button');
    b.className = `chip${active.includes(w) ? '' : ' off'}`;
    b.dataset.w = w;
    b.style.setProperty('--c', cmd.c);
    b.innerHTML = `<span class="ico">${cmd.ico}</span><span><b>${w}</b><small>${cmd.label}</small></span>`;
    b.onclick = () => execute(w, 'click');
    wrap.append(b);
  }
}

// ---------- probability bars ----------
function renderBars(probs = {}, top = null) {
  const wrap = $('bars');
  const active = activeWords();
  if (!wrap.children.length) {
    for (const w of KEYWORDS) {
      const row = document.createElement('div');
      row.className = 'bar-row'; row.dataset.w = w;
      row.style.setProperty('--c', WORD_COLORS[w]);
      row.innerHTML = `<span>${w}</span><div class="track"><div class="fill"></div></div><span class="v">—</span>`;
      wrap.append(row);
    }
  }
  for (const row of wrap.children) {
    const w = row.dataset.w, p = probs[w] ?? 0;
    row.classList.toggle('inactive', !active.includes(w));
    row.classList.toggle('top', w === top);
    row.querySelector('.fill').style.width = `${(p * 100).toFixed(1)}%`;
    row.querySelector('.v').textContent = probs[w] === undefined ? '—' : `${Math.round(p * 100)}%`;
  }
}

// ---------- spike raster (animated sweep) ----------
const raster = $('raster');
let rasterData = null, rasterStart = 0;
function drawRaster(now) {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const W = raster.clientWidth, H = raster.clientHeight;
  if (raster.width !== W * dpr) { raster.width = W * dpr; raster.height = H * dpr; }
  const g = raster.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);

  const groups = [
    { key: 'input', n: 20, color: '#ffb547' },
    { key: 'h1', n: 128, color: '#3ef2ff' },
    { key: 'h2', n: 128, color: '#ff3ea5' },
  ];
  const gap = 6, rowsTotal = 20 + 128 + 128;
  const rowH = (H - gap * 2 - 8) / rowsTotal;
  const T = rasterData?.shape.T ?? 50;
  const colW = (W - 8) / T;

  // idle: faint background noise so the panel never looks dead
  if (!rasterData) {
    g.fillStyle = 'rgba(62,242,255,0.10)';
    for (let i = 0; i < 90; i++) g.fillRect(4 + Math.random() * (W - 8), 4 + Math.random() * (H - 8), 1.5, 1.5);
    return;
  }
  const progress = Math.min(1, (now - rasterStart) / 700);
  const tMax = progress * T;
  let y0 = 4;
  const sets = [rasterData.input, rasterData.hidden[0], rasterData.hidden[1]];
  groups.forEach((grp, gi) => {
    const [ts, ns] = sets[gi] || [[], []];
    g.fillStyle = 'rgba(255,255,255,0.03)';
    g.fillRect(4, y0, W - 8, grp.n * rowH);
    g.fillStyle = grp.color;
    g.shadowColor = grp.color; g.shadowBlur = 4;
    for (let i = 0; i < ts.length; i++) {
      if (ts[i] > tMax) continue;
      g.fillRect(4 + ts[i] * colW, y0 + ns[i] * rowH, Math.max(1.2, colW * 0.7), Math.max(1, rowH * 0.9));
    }
    g.shadowBlur = 0;
    y0 += grp.n * rowH + gap;
  });
  if (progress < 1) {
    g.fillStyle = 'rgba(255,255,255,0.6)';
    g.fillRect(4 + tMax * colW, 4, 1.5, H - 8);
  }
}

// ---------- minimap ----------
function drawMinimap() {
  const c = $('minimap'); if (!c.offsetParent) return;
  const g = c.getContext('2d'), S = c.width, R = world.worldRadius, k = (S / 2 - 8) / R;
  const X = (x) => S / 2 - x * k, Z = (z) => S / 2 - z * k; // top-down, +z is "north"
  g.clearRect(0, 0, S, S);
  g.strokeStyle = 'rgba(255,62,165,0.6)'; g.lineWidth = 1.5;
  g.beginPath(); g.arc(S / 2, S / 2, R * k, 0, Math.PI * 2); g.stroke();
  g.fillStyle = 'rgba(62,242,255,0.07)';
  g.fillRect(S / 2 - 1.5 * k, 8, 3 * k, S - 16); g.fillRect(8, S / 2 - 1.5 * k, S - 16, 3 * k);
  for (const o of world.obstacles) {
    g.fillStyle = o.kind === 'planter' ? 'rgba(77,255,154,0.55)' : o.kind === 'lamp' ? 'rgba(255,181,71,0.8)' : 'rgba(138,107,255,0.8)';
    g.beginPath(); g.arc(X(o.x), Z(o.z), Math.max(2, o.r * k), 0, Math.PI * 2); g.fill();
  }
  const s = world.state;
  const px = X(s.x), pz = Z(s.z);
  if (s.lights) {
    const grad = g.createRadialGradient(px, pz, 0, px, pz, 40);
    grad.addColorStop(0, 'rgba(255,240,200,0.35)'); grad.addColorStop(1, 'rgba(255,240,200,0)');
    g.fillStyle = grad;
    g.beginPath(); g.moveTo(px, pz);
    const a = Math.atan2(-Math.cos(s.heading), -Math.sin(s.heading));
    g.arc(px, pz, 40, a - 0.45, a + 0.45); g.fill();
  }
  g.save(); g.translate(px, pz); g.rotate(-s.heading + Math.PI);
  g.fillStyle = '#3ef2ff'; g.shadowColor = '#3ef2ff'; g.shadowBlur = 10;
  g.beginPath(); g.moveTo(0, 7); g.lineTo(5, -5); g.lineTo(0, -2); g.lineTo(-5, -5); g.closePath(); g.fill();
  g.restore();
}

// ---------- telemetry ----------
function updateTelemetry() {
  const s = world.state;
  const st = $('tmState');
  if (st.className !== 'blocked' || s.moving) {
    st.textContent = s.moving ? 'DRIVING' : s.velocity > 0 ? 'BRAKING' : 'PARKED';
    st.className = s.moving ? 'moving' : '';
  } else st.textContent = 'BLOCKED';
  $('tmSpeed').innerHTML = `${s.velocity.toFixed(1)} <small>m/s</small>`;
  $('tmGear').textContent = `${s.speedLevel + 1} / 3`;
  const deg = ((-s.heading * 180 / Math.PI) % 360 + 360) % 360;
  $('tmHeading').textContent = `${String(Math.round(deg)).padStart(3, '0')}°`;
  $('tmLights').textContent = s.lights ? 'ON' : 'OFF';
  $('tmLights').className = s.lights ? 'on' : '';
}

function loop(now) {
  drawRaster(now);
  drawMinimap();
  updateTelemetry();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// ---------- level meter ----------
const levelHist = new Array(120).fill(0);
let levelThresh = 0.02;
function drawLevel(rms, thresh) {
  levelHist.push(rms); levelHist.shift(); levelThresh = thresh;
  const c = $('levelCanvas'), dpr = Math.min(window.devicePixelRatio, 2);
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const scale = (v) => Math.min(1, Math.sqrt(v / 0.25));
  const bw = W / levelHist.length;
  levelHist.forEach((v, i) => {
    const h = Math.max(1, scale(v) * (H - 4));
    g.fillStyle = v > levelThresh ? '#ff3ea5' : 'rgba(62,242,255,0.7)';
    g.fillRect(i * bw, (H - h) / 2, Math.max(1, bw - 1), h);
  });
  const ty = (H - scale(levelThresh) * (H - 4)) / 2;
  g.strokeStyle = 'rgba(255,181,71,0.6)'; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(0, ty); g.lineTo(W, ty); g.moveTo(0, H - ty); g.lineTo(W, H - ty); g.stroke(); g.setLineDash([]);
}

// ---------- mic + inference ----------
async function classify(samples) {
  const res = await fetch(`/api/predict?active=${activeWords().join(',')}`, {
    method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: samples.buffer,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function showResult(r) {
  rasterData = r.raster; rasterStart = performance.now();
  renderBars(r.probs, r.word);
  const gate = r.word === 'stop' ? ui.conf * STOP_BIAS : ui.conf;
  const accepted = r.confidence >= gate;
  const color = COMMANDS[r.word]?.c ?? '#8a93b8';
  $('heardWord').textContent = accepted ? r.word : `${r.word}?`;
  $('heardWord').className = `heard-word${accepted ? '' : ' rejected'}`;
  $('heardWord').style.color = accepted ? color : '';
  $('heardAction').textContent = accepted ? `→ ${COMMANDS[r.word].label}` : `below ${Math.round(gate * 100)}% gate — ignored`;
  $('confPct').textContent = `${Math.round(r.confidence * 100)}%`;
  $('ringBar').style.strokeDashoffset = 276.5 * (1 - r.confidence);
  $('ringBar').style.stroke = accepted ? color : '#8a93b8';
  $('latency').textContent = `${r.stats.latency_ms.toFixed(0)} ms`;
  $('stSpikes').textContent = r.stats.hidden_spikes.toLocaleString();
  $('stSparsity').textContent = `${(r.stats.sparsity * 100).toFixed(1)}%`;
  $('stSops').textContent = compact(r.stats.sops);
  $('stRatio').textContent = `${(r.stats.dense_macs / Math.max(1, r.stats.sops)).toFixed(1)}×`;
  if (accepted) execute(r.word, 'voice');
  else toast(`Didn't catch that — heard “${r.word}” at ${Math.round(r.confidence * 100)}%`, 'warn');
}

const compact = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${n}`;

const mic = new MicListener({
  onUtterance: async (samples) => {
    if (calib) return calibClip(samples);
    try { showResult(await classify(samples)); }
    catch (e) { toast(`Inference error: ${e.message}`, 'danger'); }
  },
  onLevel: drawLevel,
  onState: (st) => {
    $('calibWord').classList.toggle('hot', st === 'capturing');
    const p = $('micPill');
    p.className = `pill ${st === 'off' ? '' : st === 'processing' ? 'proc' : 'live'}`;
    p.innerHTML = `<i></i>${{ off: 'mic off', listening: 'listening', capturing: 'hearing…', processing: 'spiking…' }[st]}`;
    $('micBtn').setAttribute('aria-pressed', st !== 'off');
    $('micBtnLabel').textContent = st === 'off' ? 'Start listening' : 'Stop listening';
  },
});

async function toggleMic() {
  if (mic.state === 'off') {
    try { await mic.start(); toast('Listening — say “go”'); }
    catch (e) { toast(`Microphone unavailable: ${e.message}`, 'danger'); }
  } else mic.stop();
}

// ---------- wiring ----------
$('micBtn').onclick = toggleMic;
$('confSlider').oninput = (e) => { ui.conf = e.target.value / 100; $('confVal').textContent = `${e.target.value}%`; };
$('sensSlider').oninput = (e) => { mic.sensitivity = +e.target.value; $('sensVal').textContent = `${e.target.value}×`; };
$('speedToggle').onchange = (e) => { ui.speedWords = e.target.checked; renderChips(); renderBars(); };
$('introMic').onclick = async () => { $('intro').classList.add('hide'); await toggleMic(); };
$('introSkip').onclick = () => $('intro').classList.add('hide');

const KEYMAP = {
  w: 'go', arrowup: 'go', s: 'stop', ' ': 'stop', arrowdown: 'stop', a: 'left', arrowleft: 'left', d: 'right', arrowright: 'right',
  e: 'up', '=': 'up', '+': 'up', q: 'down', '-': 'down',
};
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.repeat) return;
  const k = e.key.toLowerCase();
  if (KEYMAP[k]) { e.preventDefault(); execute(KEYMAP[k], 'key'); }
  else if (k === 'l') execute(world.state.lights ? 'off' : 'on', 'key');
  else if (k === 'c') toast(`Camera: ${world.toggleCamera()}`);
  else if (k === 'r') { world.reset(); toast('Position reset'); }
  else if (k === 'm') toggleMic();
});

// ---------- voice calibration ----------
const CALIB_WORDS = ['go', 'stop', 'left', 'right', 'up', 'down', 'on', 'off'];
const TAKES = 5;
let calib = null; // { i: index into the word × take sequence, results: {word: [bool]} }

function calibView(which) {
  for (const id of ['calibRecord', 'calibTraining', 'calibResult']) $(id).hidden = id !== which;
  $('calibCancel').hidden = which === 'calibTraining' || which === 'calibResult';
  $('calibDone').hidden = which !== 'calibResult';
}

function renderCalib() {
  const word = CALIB_WORDS[Math.floor(calib.i / TAKES)];
  $('calibWord').textContent = word;
  $('calibTake').textContent = `take ${(calib.i % TAKES) + 1} of ${TAKES} — say it now`;
  $('calibGrid').innerHTML = CALIB_WORDS.map((w) => {
    const r = calib.results[w];
    const dots = Array.from({ length: TAKES }, (_, k) =>
      k < r.length ? `<span class="${r[k] ? 'ok' : 'no'}">${r[k] ? '✓' : '✗'}</span>` : '<span class="todo">•</span>').join('');
    return `<div class="cw${w === word ? ' now' : ''}">${w}<div class="dots">${dots}</div></div>`;
  }).join('');
}

async function startCalibration() {
  if (mic.state === 'off') {
    try { await mic.start(); } catch (e) { return toast(`Microphone unavailable: ${e.message}`, 'danger'); }
  }
  await fetch('/api/calibrate/start', { method: 'POST' });
  calib = { i: 0, results: Object.fromEntries(CALIB_WORDS.map((w) => [w, []])) };
  calibView('calibRecord');
  renderCalib();
  $('calib').classList.remove('hide');
}

async function calibClip(samples) {
  const word = CALIB_WORDS[Math.floor(calib.i / TAKES)];
  const res = await fetch(`/api/calibrate/clip?word=${word}`, { method: 'POST', body: samples.buffer }).then((r) => r.json());
  if (!calib) return; // cancelled meanwhile
  calib.results[word].push(res.base_correct);
  calib.i++;
  if (calib.i < CALIB_WORDS.length * TAKES) return renderCalib();

  calib = null;
  calibView('calibTraining');
  const report = await fetch('/api/calibrate/train', { method: 'POST' }).then((r) => r.json());
  if (report.error) { toast(report.error, 'danger'); $('calib').classList.add('hide'); return; }
  const pct = (x) => `${Math.round(x * 100)}%`;
  $('resBase').textContent = pct(report.heldout_base_acc);
  $('resPersonal').textContent = pct(report.heldout_personal_acc);
  $('resNote').textContent = `Scored on ${report.heldout_n} of your takes held out from fine-tuning. `
    + `The original model got ${pct(report.base_acc_all)} of all ${report.n_clips} takes. `
    + `The final model is trained on every take and saved for next time (${report.seconds.toFixed(0)} s).`;
  calibView('calibResult');
  refreshStatus();
}

$('calibBtn').onclick = startCalibration;
$('calibCancel').onclick = () => { calib = null; $('calib').classList.add('hide'); };
$('calibDone').onclick = () => $('calib').classList.add('hide');
$('calibReset').onclick = async () => {
  await fetch('/api/calibrate/reset', { method: 'POST' });
  toast('Back to the original model');
  refreshStatus();
};

function refreshStatus() {
  fetch('/api/status').then((r) => r.json()).then((st) => {
    const p = $('modelPill');
    if (!st.model_loaded) { p.textContent = 'untrained model — load checkpoint'; p.className = 'pill warn'; }
    else { p.textContent = `SNN ${st.hidden_sizes.join('×')} · ${st.personalized ? 'calibrated to you' : 'trained'}`; p.className = 'pill ok'; }
    $('calibReset').hidden = !st.personalized;
    $('calibBtn').textContent = st.personalized ? 'Recalibrate my voice' : 'Calibrate to my voice';
  }).catch(() => { $('modelPill').textContent = 'backend offline'; $('modelPill').className = 'pill warn'; });
}
refreshStatus();

renderChips();
renderBars();

// handy for demos / debugging from the browser console
window.neurochair = { world, mic, execute, classify, showResult, calibClip, calibView };
