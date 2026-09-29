import { createScene } from './scene.js';
import { MicListener } from './audio.js';

const COMMANDS = {
  go:    { label: 'Drive forward', key: 'W', c: '#2f58c9' },
  stop:  { label: 'Brake',         key: 'S', c: '#d6456f' },
  left:  { label: 'Turn left',     key: 'A', c: '#1f1d36' },
  right: { label: 'Turn right',    key: 'D', c: '#1f1d36' },
  up:    { label: 'Faster',        key: 'E', c: '#de9a2e' },
  down:  { label: 'Slower',        key: 'Q', c: '#de9a2e' },
  on:    { label: 'Lights on',     key: 'L', c: '#de9a2e' },
  off:   { label: 'Lights off',    key: 'L', c: '#de9a2e' },
};
const INK = '#1f1d36', INK_2 = '#5e5a78', RULE = '#e6ddd8', ROSA = '#d6456f', OCHRE = '#de9a2e', CANVAS_FONT = '"Atkinson Hyperlegible Next", sans-serif';
const STOP_BIAS = 0.6; // "stop" only needs 60% of the normal confidence gate

const $ = (id) => document.getElementById(id);
const ui = { conf: 0.45, speedWords: true };

// write to the DOM only when the text actually changes
const shown = new Map();
function setText(id, text) {
  if (shown.get(id) === text) return;
  shown.set(id, text);
  $(id).textContent = text;
}

// size a canvas to its CSS box at device resolution; returns [ctx, width, height]
function fit(c) {
  const dpr = Math.min(window.devicePixelRatio, 2), w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}

// ---------- toast ----------
let toastTimer;
function toast(msg, kind = '') {
  const t = $('toast');
  t.textContent = msg; t.className = `toast show ${kind}`;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = 'toast'), 2600);
}

// ---------- scene ----------
let blocked = false;
const world = createScene($('world'), {
  onBlocked: (what) => {
    blocked = true;
    toast(`Stopped: ${what === 'boundary' ? 'edge of the courtyard' : 'something ahead'}`, 'danger');
  },
});

// keep the chair centred in the part of the scene the deck doesn't cover
new ResizeObserver(([entry]) => world.setInset(entry.target.offsetHeight + 16)).observe(document.querySelector('.deck'));

// ---------- commands ----------
function activeWords() {
  const words = ['go', 'stop', 'left', 'right', 'on', 'off'];
  if (ui.speedWords) words.push('up', 'down');
  return words;
}

const history = [];
function execute(word, source = 'voice') {
  const actions = {
    go: () => world.go(), stop: () => world.stop(),
    left: () => world.turn('left'), right: () => world.turn('right'),
    up: () => world.speed(+1), down: () => world.speed(-1),
    on: () => world.setLights(true), off: () => world.setLights(false),
  };
  if (!actions[word]) return;
  actions[word]();
  if (word === 'go' || word === 'stop') blocked = false;
  world.burst(COMMANDS[word].c);
  history.unshift(source === 'voice' ? word : `${word} (key)`);
  history.length = Math.min(history.length, 5);
  $('log').textContent = `Recent: ${history.join(' · ')}`;
  const key = document.querySelector(`.key[data-w="${word}"]`);
  if (key) { key.classList.add('flash'); setTimeout(() => key.classList.remove('flash'), 600); }
}

// ---------- command keys (with probability fill) ----------
function buildKeys() {
  const wrap = $('keys');
  for (const [w, cmd] of Object.entries(COMMANDS)) {
    const b = document.createElement('button');
    b.className = 'key'; b.dataset.w = w;
    b.style.setProperty('--c', cmd.c);
    b.setAttribute('aria-label', `${w}: ${cmd.label}`);
    b.title = cmd.label;
    b.innerHTML = `<b>${w}</b><small><span>${cmd.key}</span><span class="p"></span></small><i class="fill"></i>`;
    b.onclick = () => execute(w, 'click');
    wrap.append(b);
  }
  renderKeys();
}

function renderKeys(probs = null, top = null) {
  const active = activeWords();
  for (const b of $('keys').children) {
    const w = b.dataset.w, p = probs?.[w] ?? 0;
    b.classList.toggle('off', !active.includes(w));
    b.classList.toggle('top', w === top);
    b.querySelector('.fill').style.width = `${(p * 100).toFixed(1)}%`;
    b.querySelector('.p').textContent = probs ? `${Math.round(p * 100)}%` : '';
  }
}

// ---------- spike raster: drawn only while its sweep animates ----------
const raster = $('raster');
let rasterData = null, rasterStart = 0, rasterDirty = true;
function drawRaster(now) {
  const [g, W, H] = fit(raster);
  if (!W) return false;
  if (!rasterData) {
    g.fillStyle = INK_2; g.font = `13px ${CANVAS_FONT}`; g.textAlign = 'center';
    g.fillText('Say a command to see which neurons fire.', W / 2, H / 2 + 4);
    return false;
  }
  const { shape } = rasterData;
  const groups = [{ n: shape.input, color: OCHRE, data: rasterData.input },
    ...shape.hidden.map((n, i) => ({ n, color: i % 2 ? ROSA : INK, data: rasterData.hidden[i] }))];
  const pad = 6, gap = 6, T = shape.T;
  const rowH = (H - pad * 2 - gap * (groups.length - 1)) / groups.reduce((sum, grp) => sum + grp.n, 0);
  const colW = (W - pad * 2) / T;
  g.fillStyle = RULE;
  for (let t = 10; t < T; t += 10) g.fillRect(pad + t * colW, pad, 1, H - pad * 2);
  const progress = Math.min(1, (now - rasterStart) / 700), tMax = progress * T;
  let y0 = pad;
  for (const grp of groups) {
    const [ts, ns] = grp.data;
    g.fillStyle = grp.color;
    const w = Math.max(1.5, colW * 0.6), h = Math.max(1.2, rowH * 0.85);
    for (let i = 0; i < ts.length; i++) if (ts[i] <= tMax) g.fillRect(pad + ts[i] * colW, y0 + ns[i] * rowH, w, h);
    y0 += grp.n * rowH + gap;
    if (y0 < H - pad) { g.fillStyle = RULE; g.fillRect(pad, y0 - gap / 2, W - pad * 2, 1); }
  }
  if (progress < 1) { g.fillStyle = INK; g.fillRect(pad + tMax * colW, pad, 1.5, H - pad * 2); }
  return progress < 1;
}

// ---------- plan view: redrawn only when the chair moves ----------
const minimap = $('minimap');
let mapKey = '';
const hex = (n) => `#${n.toString(16).padStart(6, '0')}`;
function drawMinimap() {
  const s = world.state;
  const key = `${s.x.toFixed(2)},${s.z.toFixed(2)},${s.heading.toFixed(3)},${s.lights},${minimap.clientWidth}`;
  if (key === mapKey) return;
  mapKey = key;
  const [g, S] = fit(minimap);
  if (!S) return;
  const R = world.worldRadius, k = (S / 2 - 8) / R;
  const X = (x) => S / 2 - x * k, Z = (z) => S / 2 - z * k; // top-down, +z is "north"
  g.fillStyle = '#efe7e3';
  g.fillRect(S / 2 - 1.5 * k, 8, 3 * k, S - 16); g.fillRect(8, S / 2 - 1.5 * k, S - 16, 3 * k);
  g.strokeStyle = ROSA; g.lineWidth = 2;
  g.beginPath(); g.arc(S / 2, S / 2, R * k + 1, 0, Math.PI * 2); g.stroke();
  for (const o of world.obstacles) {
    if (o.kind === 'column') { g.fillStyle = hex(o.color); g.fillRect(X(o.x) - k / 2, Z(o.z) - k / 2, k, k); }
    else if (o.kind === 'tree') { g.strokeStyle = hex(o.color); g.lineWidth = 1.5; g.beginPath(); g.arc(X(o.x), Z(o.z), o.r * k, 0, Math.PI * 2); g.stroke(); }
    else { g.fillStyle = INK; g.beginPath(); g.arc(X(o.x), Z(o.z), 1.5, 0, Math.PI * 2); g.fill(); }
  }
  const px = X(s.x), pz = Z(s.z);
  if (s.lights) {
    const a = Math.atan2(-Math.cos(s.heading), -Math.sin(s.heading));
    g.fillStyle = 'rgba(222,154,46,0.35)';
    g.beginPath(); g.moveTo(px, pz); g.arc(px, pz, 30, a - 0.45, a + 0.45); g.fill();
  }
  g.save(); g.translate(px, pz); g.rotate(-s.heading + Math.PI);
  g.fillStyle = INK;
  g.beginPath(); g.moveTo(0, 7); g.lineTo(5, -5); g.lineTo(0, -2); g.lineTo(-5, -5); g.closePath(); g.fill();
  g.restore();
}

// ---------- telemetry ----------
function updateTelemetry() {
  const s = world.state;
  if (s.moving) blocked = false;
  const state = s.moving ? 'Driving' : s.velocity > 0 ? 'Braking' : blocked ? 'Blocked' : 'Parked';
  setText('tmState', state);
  $('tmState').className = s.moving ? 'moving' : blocked ? 'blocked' : '';
  setText('tmSpeed', `${s.velocity.toFixed(1)} m/s`);
  setText('tmGear', `${s.speedLevel + 1}`);
  const deg = ((-s.heading * 180 / Math.PI) % 360 + 360) % 360;
  setText('tmHeading', `${String(Math.round(deg) % 360).padStart(3, '0')}°`);
  setText('tmLights', s.lights ? 'on' : 'off');
}

// ---------- level meter ----------
const levelHist = new Array(90).fill(0);
let levelThresh = 0.02, levelDirty = true;
function onLevel(rms, thresh) { levelHist.push(rms); levelHist.shift(); levelThresh = thresh; levelDirty = true; }
function drawLevel() {
  const [g, W, H] = fit($('levelCanvas'));
  if (!W) return;
  const scale = (v) => Math.min(1, Math.sqrt(v / 0.25));
  const bw = W / levelHist.length;
  levelHist.forEach((v, i) => {
    const h = Math.max(1, scale(v) * (H - 6));
    g.fillStyle = v > levelThresh ? ROSA : INK_2;
    g.fillRect(i * bw, (H - h) / 2, Math.max(1, bw - 1), h);
  });
  const ty = (H - scale(levelThresh) * (H - 6)) / 2;
  g.strokeStyle = OCHRE; g.lineWidth = 1.5; g.setLineDash([4, 3]);
  g.beginPath(); g.moveTo(0, ty); g.lineTo(W, ty); g.moveTo(0, H - ty); g.lineTo(W, H - ty); g.stroke(); g.setLineDash([]);
}

// ---------- one UI loop for everything outside the 3D scene ----------
let rasterAnimating = false;
function tick(now) {
  updateTelemetry();
  drawMinimap();
  if (rasterAnimating || rasterDirty) { rasterAnimating = drawRaster(now); rasterDirty = false; }
  if (levelDirty) { drawLevel(); levelDirty = false; }
  requestAnimationFrame(tick);
}
window.addEventListener('resize', () => { rasterDirty = true; levelDirty = true; mapKey = ''; });

// ---------- mic + inference ----------
async function classify(samples) {
  const res = await fetch(`/api/predict?active=${activeWords().join(',')}`, {
    method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: samples.buffer,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const compact = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${n}`;

function showResult(r) {
  rasterData = r.raster; rasterStart = performance.now(); rasterAnimating = true;
  renderKeys(r.probs, r.word);
  const gate = r.word === 'stop' ? ui.conf * STOP_BIAS : ui.conf;
  const accepted = r.confidence >= gate;
  const word = $('heardWord');
  word.textContent = accepted ? r.word : `${r.word}?`;
  word.className = `word ${accepted ? 'accepted' : 'rejected'}`;
  word.style.fontStretch = `${Math.round(50 + 100 * r.confidence)}%`; // letter width = confidence
  $('heardAction').textContent = accepted
    ? COMMANDS[r.word].label
    : `Not sure enough (needs ${Math.round(gate * 100)}%), so nothing happened.`;
  $('confPct').textContent = `${Math.round(r.confidence * 100)}% sure`;
  $('latency').textContent = `${r.stats.latency_ms.toFixed(0)} ms`;
  $('stSpikes').textContent = r.stats.hidden_spikes.toLocaleString();
  $('stSparsity').textContent = `${(r.stats.sparsity * 100).toFixed(1)}%`;
  $('stSops').textContent = compact(r.stats.sops);
  $('stRatio').textContent = `${(r.stats.dense_macs / Math.max(1, r.stats.sops)).toFixed(1)}×`;
  if (accepted) execute(r.word, 'voice');
}

let lastUtterance = null; // kept so Shift+key can relabel it

async function correct(word) {
  if (!lastUtterance) return toast('Nothing to correct yet. Say a command first.', 'warn');
  const res = await fetch(`/api/correct?word=${word}`, { method: 'POST', body: lastUtterance.buffer }).then((r) => r.json());
  lastUtterance = null;
  toast(`Saved that clip as “${word}”. ${res.corrections} correction(s) ready to retrain.`);
  refreshStatus();
}

const mic = new MicListener({
  onUtterance: async (samples) => {
    if (calib) return calibClip(samples);
    lastUtterance = samples;
    try { showResult(await classify(samples)); }
    catch (e) { toast(`Couldn’t reach the model: ${e.message}`, 'danger'); }
  },
  onLevel,
  onState: (st) => {
    $('calibWord').classList.toggle('hot', st === 'capturing');
    const p = $('micPill');
    p.className = `tag ${st === 'off' ? '' : st === 'processing' ? 'proc' : 'live'}`;
    p.innerHTML = `<i></i>${{ off: 'Mic off', listening: 'Listening', capturing: 'Hearing', processing: 'Thinking' }[st]}`;
    $('micBtn').setAttribute('aria-pressed', st !== 'off');
    $('micBtnLabel').textContent = st === 'off' ? 'Start listening' : 'Stop listening';
  },
});

async function toggleMic() {
  if (mic.state === 'off') {
    try { await mic.start(); toast('Listening. Say “go” to start.'); }
    catch (e) { toast(`Microphone blocked: ${e.message}. Allow it in the browser’s site settings.`, 'danger'); }
  } else mic.stop();
}

// ---------- wiring ----------
function toggleDrawer(open = $('tune').hidden) {
  $('tune').hidden = !open;
  $('tuneBtn').setAttribute('aria-expanded', open);
}
$('tuneBtn').onclick = () => toggleDrawer();
$('micBtn').onclick = toggleMic;
$('confSlider').oninput = (e) => { ui.conf = e.target.value / 100; $('confVal').textContent = `${e.target.value}%`; };
$('sensSlider').oninput = (e) => { mic.sensitivity = +e.target.value; $('sensVal').textContent = `${e.target.value}×`; };
$('speedToggle').onchange = (e) => { ui.speedWords = e.target.checked; renderKeys(); };
$('introMic').onclick = async () => { $('intro').classList.add('hide'); await toggleMic(); };
$('introSkip').onclick = () => $('intro').classList.add('hide');

const KEYMAP = {
  w: 'go', arrowup: 'go', s: 'stop', ' ': 'stop', arrowdown: 'stop', a: 'left', arrowleft: 'left', d: 'right', arrowright: 'right',
  e: 'up', '=': 'up', '+': 'up', q: 'down', '-': 'down',
};
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.repeat) return;
  if (e.key === ' ' && e.target.tagName === 'BUTTON') return; // let Space press the focused button
  const k = e.key.toLowerCase();
  const word = KEYMAP[k] ?? (k === 'l' ? (world.state.lights ? 'off' : 'on') : null);
  if (word) {
    e.preventDefault();
    if (e.shiftKey) correct(word);
    execute(word, 'key');
  } else if (k === 'escape') toggleDrawer(false);
  else if (k === 'c') toast(`Camera: ${world.toggleCamera() === 'orbit' ? 'free (drag to look around)' : 'follow'}`);
  else if (k === 'r') { world.reset(); toast('Back at the start'); }
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
  $('calibTake').textContent = `take ${(calib.i % TAKES) + 1} of ${TAKES}`;
  $('calibGrid').innerHTML = CALIB_WORDS.map((w) => {
    const r = calib.results[w];
    const dots = Array.from({ length: TAKES }, (_, k) =>
      k < r.length ? `<span class="${r[k] ? 'ok' : 'no'}">${r[k] ? '✓' : '✗'}</span>` : '<span class="todo">●</span>').join('');
    return `<div class="cw${w === word ? ' now' : ''}">${w}<div class="dots">${dots}</div></div>`;
  }).join('');
}

async function startCalibration() {
  if (mic.state === 'off') {
    try { await mic.start(); } catch (e) { return toast(`Microphone blocked: ${e.message}`, 'danger'); }
  }
  toggleDrawer(false);
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
  await train();
}

async function train() {
  toggleDrawer(false);
  calibView('calibTraining');
  $('calib').classList.remove('hide');
  const report = await fetch('/api/calibrate/train', { method: 'POST' }).then((r) => r.json());
  if (report.error) { toast(report.error, 'danger'); $('calib').classList.add('hide'); return; }
  const pct = (x) => `${Math.round(x * 100)}%`;
  $('resBase').textContent = pct(report.heldout_base_acc);
  $('resPersonal').textContent = pct(report.heldout_personal_acc);
  $('resNote').textContent = `Scored on ${report.heldout_n} of your takes that were held out from fine-tuning. `
    + `The original model got ${pct(report.base_acc_all)} of all ${report.n_clips} takes. `
    + `The saved model is trained on every take (${report.seconds.toFixed(0)} s).`;
  calibView('calibResult');
  refreshStatus();
}

$('calibBtn').onclick = startCalibration;
$('calibRetrain').onclick = train;
$('calibCancel').onclick = () => { calib = null; $('calib').classList.add('hide'); };
$('calibDone').onclick = () => $('calib').classList.add('hide');
$('calibReset').onclick = async () => {
  await fetch('/api/calibrate/reset', { method: 'POST' });
  toast('Back to the base model');
  refreshStatus();
};

function refreshStatus() {
  fetch('/api/status').then((r) => r.json()).then((st) => {
    const p = $('modelPill');
    if (!st.model_loaded) { p.textContent = 'Untrained model'; p.className = 'tag warn'; p.title = 'Put snn_checkpoint.pt in checkpoints/ and restart the server'; }
    else { p.textContent = `${st.personalized ? 'Calibrated to you' : 'Base model'} · ${st.hidden_sizes.join('×')}`; p.className = 'tag'; p.title = ''; }
    $('calibReset').hidden = !st.personalized;
    $('calibRetrain').hidden = !st.corrections;
    $('calibRetrain').textContent = `Retrain with ${st.corrections} correction(s)`;
    $('calibBtn').textContent = st.personalized ? 'Recalibrate my voice' : 'Calibrate to my voice';
  }).catch(() => { $('modelPill').textContent = 'Server offline'; $('modelPill').className = 'tag warn'; });
}

buildKeys();
refreshStatus();
requestAnimationFrame(tick);

// handy for demos / debugging from the browser console
window.neurochair = { world, mic, execute, classify, showResult, calibClip, calibView };
