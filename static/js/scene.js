// NeuroChair 3D scene: a procedurally-built wheelchair in a sunlit plaster courtyard.
// Everything is made from Three.js primitives -- no model files to download.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const WORLD_RADIUS = 28;
const CHAIR_RADIUS = 0.45;
const SPEEDS = [0.9, 1.6, 2.4];          // m/s for speed levels 1..3
const TURN_RATE = 2.2;                   // rad/s

export const PALETTE = {
  ground: 0xe7ded8, paving: 0xd3c7c0, plaster: 0xf3ece9, ink: 0x1f1d36,
  rosa: 0xd6456f, ochre: 0xde9a2e, cobalt: 0x2f58c9, lavender: 0xa99bd6, leaf: 0x6d8b4e,
};

export function createScene(canvas, { onBlocked } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.NeutralToneMapping;

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xf0ddd6, 35, 95);

  const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 200);
  camera.position.set(0, 3, -7);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.47;
  controls.minDistance = 2;
  controls.maxDistance = 30;
  controls.enabled = false;

  // ---------- light: sky fill + one low sun whose shadow box follows the chair ----------
  scene.add(new THREE.HemisphereLight(0xe4ecff, 0xd9b3a4, 1.6));
  const sun = new THREE.DirectionalLight(0xfff0de, 2.4);
  const SUN_OFFSET = new THREE.Vector3(-10, 14, -7);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.bias = -0.0005;
  Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12, near: 1, far: 50 });
  scene.add(sun, sun.target);

  // ---------- sky ----------
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(120, 24, 12),
    new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      vertexShader: `varying float vY; void main(){ vY = normalize(position).y; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `varying float vY; void main(){
        vec3 horizon = vec3(0.94,0.86,0.83); vec3 zenith = vec3(0.55,0.68,0.88);
        gl_FragColor = vec4(mix(horizon, zenith, smoothstep(0.0, 0.6, vY)), 1.0); }`,
    }),
  ));

  // ---------- ground, paving joints, walkways, perimeter wall ----------
  const lambert = (color, extra = {}) => new THREE.MeshLambertMaterial({ color, ...extra });
  const ground = new THREE.Mesh(new THREE.CircleGeometry(100, 48), lambert(PALETTE.ground));
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const joints = new THREE.GridHelper(WORLD_RADIUS * 2, WORLD_RADIUS, PALETTE.paving, PALETTE.paving);
  joints.position.y = 0.003;
  scene.add(joints);

  const walkMat = lambert(PALETTE.plaster);
  for (const rot of [0, Math.PI / 2]) {
    const p = new THREE.Mesh(new THREE.PlaneGeometry(3, WORLD_RADIUS * 2), walkMat);
    p.rotation.set(-Math.PI / 2, 0, rot); p.position.y = 0.005; p.receiveShadow = true;
    scene.add(p);
  }

  const perimeter = new THREE.Mesh(
    new THREE.CylinderGeometry(WORLD_RADIUS + 0.6, WORLD_RADIUS + 0.6, 0.8, 96, 1, true),
    lambert(PALETTE.rosa, { side: THREE.DoubleSide }),
  );
  perimeter.position.y = 0.4; perimeter.receiveShadow = true;
  scene.add(perimeter);

  // tall coloured walls outside the perimeter -- scenery only, no collision
  for (const [w, h, x, z, rot, color] of [
    [18, 7, 0, 36, 0, PALETTE.rosa], [12, 10, -34, 12, 1.2, PALETTE.ochre],
    [14, 6, 30, -22, -0.9, PALETTE.lavender], [4, 13, -20, -33, 0.4, PALETTE.cobalt], [9, 4.5, 26, 24, 0.7, PALETTE.ochre],
  ]) {
    const wall = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.8), lambert(color));
    wall.position.set(x, h / 2, z); wall.rotation.y = rot; wall.receiveShadow = true;
    scene.add(wall);
  }

  // ---------- obstacles ----------
  const obstacles = []; // {x, z, r, kind, color}
  const colGeo = new THREE.BoxGeometry(1, 4, 1);
  const colColors = [PALETTE.rosa, PALETTE.ochre, PALETTE.lavender, PALETTE.cobalt];
  [[-6, -6], [6, -6], [-6, 6], [6, 6], [-14, 0], [14, 0], [0, -14], [0, 14]].forEach(([x, z], i) => {
    const color = colColors[i % colColors.length];
    const col = new THREE.Mesh(colGeo, lambert(color));
    col.position.set(x, 2, z); col.castShadow = true; col.receiveShadow = true;
    scene.add(col);
    obstacles.push({ x, z, r: 0.7, kind: 'column', color });
  });

  const planterGeo = new THREE.BoxGeometry(1.4, 0.6, 1.4);
  const trunkGeo = new THREE.CylinderGeometry(0.07, 0.1, 1.2, 8);
  const canopyGeo = new THREE.IcosahedronGeometry(0.85, 1);
  const planterMat = lambert(PALETTE.plaster), trunkMat = lambert(0x7a5a44), leafMat = lambert(PALETTE.leaf, { flatShading: true });
  for (const [x, z] of [[-10, -10], [10, 10], [-10, 10], [10, -10], [-18, 8], [18, -8], [8, 18], [-8, -18]]) {
    const g = new THREE.Group();
    const box = new THREE.Mesh(planterGeo, planterMat); box.position.y = 0.3;
    const trunk = new THREE.Mesh(trunkGeo, trunkMat); trunk.position.y = 1.1;
    const canopy = new THREE.Mesh(canopyGeo, leafMat); canopy.position.y = 2.1;
    for (const m of [box, trunk, canopy]) { m.castShadow = true; m.receiveShadow = true; g.add(m); }
    g.position.set(x, 0, z);
    scene.add(g);
    obstacles.push({ x, z, r: 0.95, kind: 'tree', color: PALETTE.leaf });
  }

  const bollardGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.9, 12);
  const capGeo = new THREE.CylinderGeometry(0.125, 0.125, 0.08, 12);
  const bollardMat = lambert(PALETTE.ink), capMat = lambert(PALETTE.ochre);
  for (const [x, z] of [[-2.4, 9], [2.4, -9], [9, 2.4], [-9, -2.4], [-2.4, -20], [2.4, 20]]) {
    const b = new THREE.Mesh(bollardGeo, bollardMat); b.position.set(x, 0.45, z); b.castShadow = true;
    const cap = new THREE.Mesh(capGeo, capMat); cap.position.set(x, 0.94, z);
    scene.add(b, cap);
    obstacles.push({ x, z, r: 0.15, kind: 'bollard', color: PALETTE.ink });
  }

  // ---------- wheelchair ----------
  const chair = buildWheelchair();
  scene.add(chair.group);

  // expanding ground ring on each recognised command
  const effects = [];
  const ringGeo = new THREE.RingGeometry(0.5, 0.56, 64);

  // ---------- motion state ----------
  const s = {
    x: 0, z: -3, heading: 0, targetHeading: 0,
    moving: false, speedLevel: 1, velocity: 0,
    lights: false, cameraMode: 'chase', blockedCooldown: 0,
  };

  let dirty = true; // render only when something on screen changes
  controls.addEventListener('change', () => { dirty = true; });

  let inset = 0; // px hidden behind the bottom panel; the view is re-centred on what stays visible
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / (h + inset);
    camera.setViewOffset(w, h + inset, 0, inset, w, h);
    dirty = true;
  }
  window.addEventListener('resize', resize);
  resize();

  const fwd = new THREE.Vector3();
  const setFwd = () => fwd.set(Math.sin(s.heading), 0, Math.cos(s.heading));

  function collides(x, z) {
    if (Math.hypot(x, z) > WORLD_RADIUS - CHAIR_RADIUS) return 'boundary';
    for (const o of obstacles) if (Math.hypot(x - o.x, z - o.z) < o.r + CHAIR_RADIUS) return o.kind;
    return null;
  }

  function burst(color) {
    const ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.position.set(s.x, 0.02, s.z);
    scene.add(ring); effects.push({ mesh: ring, t: 0 });
    chair.ledFlash = 1;
    dirty = true;
  }

  // ---------- public controls ----------
  const api = {
    state: s,
    obstacles,
    worldRadius: WORLD_RADIUS,
    go() { s.moving = true; },
    stop() { s.moving = false; },
    turn(dir) { s.targetHeading += dir === 'left' ? Math.PI / 2 : -Math.PI / 2; },
    speed(delta) { s.speedLevel = Math.max(0, Math.min(SPEEDS.length - 1, s.speedLevel + delta)); },
    setLights(on) { s.lights = on; },
    burst,
    setInset(px) { inset = px; resize(); },
    toggleCamera() {
      s.cameraMode = s.cameraMode === 'chase' ? 'orbit' : 'chase';
      controls.enabled = s.cameraMode === 'orbit';
      dirty = true;
      return s.cameraMode;
    },
    reset() { Object.assign(s, { x: 0, z: -3, heading: 0, targetHeading: 0, moving: false, velocity: 0 }); dirty = true; },
    speedMps: () => SPEEDS[s.speedLevel],
  };

  // ---------- main loop ----------
  const clock = new THREE.Clock();
  const desired = new THREE.Vector3();
  function frame() {
    requestAnimationFrame(frame);
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.elapsedTime;
    let changed = dirty;

    // heading: smooth rotation toward target
    const dh = s.targetHeading - s.heading;
    const turnStep = Math.sign(dh) * Math.min(Math.abs(dh), TURN_RATE * dt);
    s.heading += turnStep;

    // velocity: ease toward target speed
    const target = s.moving ? SPEEDS[s.speedLevel] : 0;
    s.velocity += (target - s.velocity) * Math.min(1, dt * (s.moving ? 2.5 : 5));
    if (Math.abs(s.velocity) < 0.005) s.velocity = 0;
    setFwd();

    if (s.velocity > 0) {
      const nx = s.x + fwd.x * s.velocity * dt, nz = s.z + fwd.z * s.velocity * dt;
      // proximity sensor looks a bit ahead so it brakes before touching
      const hit = collides(s.x + fwd.x * 0.35, s.z + fwd.z * 0.35) && collides(nx + fwd.x * 0.3, nz + fwd.z * 0.3);
      if (hit) {
        s.moving = false; s.velocity = 0;
        if (s.blockedCooldown <= 0) { onBlocked?.(hit); burst(PALETTE.rosa); s.blockedCooldown = 1.2; }
      } else { s.x = nx; s.z = nz; }
    }
    s.blockedCooldown -= dt;
    if (s.velocity > 0 || turnStep !== 0) changed = true;

    // place chair; wheels roll forward and counter-rotate when turning in place
    chair.group.position.set(s.x, s.velocity > 0 ? Math.sin(t * 18) * 0.004 * s.velocity : 0, s.z);
    chair.group.rotation.y = s.heading;
    const roll = s.velocity * dt / 0.3, spin = (turnStep * 0.33) / 0.3;
    chair.wheelL.rotation.x += roll - spin;
    chair.wheelR.rotation.x += roll + spin;
    for (const c of chair.casters) {
      c.wheel.rotation.x += s.velocity * dt / 0.08;
      const fork = (Math.abs(dh) > 0.01 ? Math.sign(dh) * 0.6 : 0) - c.fork.rotation.y;
      c.fork.rotation.y += fork * Math.min(1, dt * 6);
      if (Math.abs(fork) > 0.001) changed = true;
    }

    // headlights + status LED
    const lf = s.lights ? 1 : 0;
    if (Math.abs(lf - chair.lightLevel) > 0.001) {
      chair.lightLevel += (lf - chair.lightLevel) * Math.min(1, dt * 10);
      const l = chair.lightLevel;
      chair.headMat.color.setRGB(0.35 + 0.65 * l, 0.35 + 0.6 * l, 0.38 + 0.4 * l);
      changed = true;
    }
    if (chair.ledFlash > 0) {
      chair.ledFlash = Math.max(0, chair.ledFlash - dt * 2);
      chair.ledMat.color.setHex(PALETTE.ink).lerp(chair.ledOn, chair.ledFlash);
      changed = true;
    }

    for (let i = effects.length - 1; i >= 0; i--) {
      const e = effects[i]; e.t += dt;
      const k = e.t / 1.0;
      e.mesh.scale.setScalar(1 + k * 6);
      e.mesh.material.opacity = 0.8 * Math.max(0, 1 - k);
      if (k >= 1) { scene.remove(e.mesh); e.mesh.material.dispose(); effects.splice(i, 1); }
      changed = true;
    }

    // sun + shadow box follow the chair
    sun.target.position.set(s.x, 0, s.z);
    sun.position.copy(sun.target.position).add(SUN_OFFSET);

    if (s.cameraMode === 'chase') {
      desired.set(s.x - fwd.x * 5 + fwd.z * 0.9, 2.3, s.z - fwd.z * 5 - fwd.x * 0.9);
      if (camera.position.distanceToSquared(desired) > 1e-6) {
        camera.position.lerp(desired, Math.min(1, dt * 3));
        changed = true;
      }
      camera.lookAt(s.x + fwd.x * 1.5, 0.9, s.z + fwd.z * 1.5);
    } else {
      controls.target.lerp(desired.set(s.x, 0.6, s.z), Math.min(1, dt * 4));
      controls.update();
    }

    if (changed) { renderer.render(scene, camera); dirty = false; }
  }
  requestAnimationFrame(frame);

  return api;
}

function buildWheelchair() {
  const group = new THREE.Group();
  const frameMat = new THREE.MeshStandardMaterial({ color: PALETTE.cobalt, roughness: 0.4, metalness: 0.35 });
  const metal = new THREE.MeshStandardMaterial({ color: 0xc9ccd6, roughness: 0.3, metalness: 0.8 });
  const dark = new THREE.MeshLambertMaterial({ color: PALETTE.ink });
  const fabric = new THREE.MeshLambertMaterial({ color: 0x2b2945 });
  const tire = new THREE.MeshLambertMaterial({ color: 0x18171f });

  const cast = (m) => { m.castShadow = true; return m; };
  const tube = (a, b, r = 0.016, mat = frameMat) => {
    const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
    const m = cast(new THREE.Mesh(new THREE.CylinderGeometry(r, r, va.distanceTo(vb), 10), mat));
    m.position.copy(va).add(vb).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
    group.add(m); return m;
  };

  // big rear wheels
  const makeWheel = (side) => {
    const w = new THREE.Group();
    const t = cast(new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.03, 12, 40), tire));
    t.rotation.y = Math.PI / 2; w.add(t);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(0.27, 0.01, 6, 40), metal);
    rim.rotation.y = Math.PI / 2; w.add(rim);
    const push = new THREE.Mesh(new THREE.TorusGeometry(0.26, 0.009, 6, 40), metal);
    push.rotation.y = Math.PI / 2; push.position.x = side * 0.045; w.add(push);
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const sp = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 0.27, 4), metal);
      sp.position.set(0, Math.cos(a) * 0.135, Math.sin(a) * 0.135);
      sp.rotation.x = -a;
      w.add(sp);
    }
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.06, 12), dark);
    hub.rotation.z = Math.PI / 2; w.add(hub);
    w.position.set(side * 0.31, 0.3, -0.1);
    group.add(w);
    return w;
  };
  const wheelL = makeWheel(1), wheelR = makeWheel(-1);

  // front casters
  const casters = [];
  for (const side of [1, -1]) {
    const fork = new THREE.Group();
    fork.position.set(side * 0.21, 0.2, 0.33);
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.08, 8), metal);
    stem.position.y = -0.02; fork.add(stem);
    const wheel = new THREE.Group();
    const cw = cast(new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.02, 8, 20), tire));
    cw.rotation.y = Math.PI / 2; wheel.add(cw);
    wheel.position.set(0, -0.12, -0.03);
    fork.add(wheel);
    group.add(fork);
    casters.push({ fork, wheel });
  }

  // frame
  for (const x of [0.22, -0.22]) {
    tube([x, 0.46, -0.2], [x, 0.46, 0.3]);          // seat rail
    tube([x, 0.46, 0.3], [x, 0.14, 0.44]);          // front leg to footrest
    tube([x, 0.2, 0.33], [x, 0.46, 0.3]);           // caster mount
    tube([x, 0.3, -0.1], [x, 0.46, -0.15]);         // axle strut
    tube([x, 0.46, -0.2], [x, 1.02, -0.26]);        // back post
    tube([x, 1.02, -0.26], [x, 1.02, -0.36], 0.018, dark); // push handle
    tube([x, 0.46, 0.05], [x, 0.66, 0.05]);         // armrest post
  }
  tube([0.22, 0.46, 0.25], [-0.22, 0.46, 0.25]);

  const seat = cast(new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.07, 0.46), fabric));
  seat.position.set(0, 0.5, 0.05); group.add(seat);
  const back = cast(new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.46, 0.05), fabric));
  back.position.set(0, 0.78, -0.225); back.rotation.x = -0.1; group.add(back);
  for (const x of [0.25, -0.25]) {
    const arm = cast(new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.04, 0.34), dark));
    arm.position.set(x, 0.68, 0.02); group.add(arm);
  }
  const foot = cast(new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.02, 0.14), dark));
  foot.position.set(0, 0.14, 0.46); group.add(foot);
  const pack = cast(new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.12, 0.24), dark));
  pack.position.set(0, 0.36, -0.02); group.add(pack);

  // controller on the right armrest; its LED flashes when a command is accepted
  const ctrl = cast(new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.05, 0.1), dark));
  ctrl.position.set(-0.25, 0.725, 0.14); group.add(ctrl);
  const ledMat = new THREE.MeshBasicMaterial({ color: PALETTE.ink });
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.016, 10, 8), ledMat);
  led.position.set(-0.25, 0.76, 0.17); group.add(led);

  // headlights on the armrests
  const headMat = new THREE.MeshBasicMaterial({ color: 0x5a5a62 });
  for (const x of [0.25, -0.25]) {
    const h = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.02, 12), headMat);
    h.rotation.x = Math.PI / 2; h.position.set(x, 0.68, 0.2); group.add(h);
  }

  // a simple seated rider so the chair reads at a distance
  const rider = new THREE.MeshLambertMaterial({ color: 0x8a7fb8 });
  const skin = new THREE.MeshLambertMaterial({ color: 0xc99a7c });
  const torso = cast(new THREE.Mesh(new THREE.CapsuleGeometry(0.15, 0.3, 4, 10), rider));
  torso.position.set(0, 0.86, -0.1); torso.rotation.x = -0.08; group.add(torso);
  const head = cast(new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 12), skin));
  head.position.set(0, 1.2, -0.08); group.add(head);
  for (const x of [0.09, -0.09]) {
    const thigh = cast(new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.3, 4, 8), rider));
    thigh.rotation.x = Math.PI / 2; thigh.position.set(x, 0.6, 0.12); group.add(thigh);
    const shin = cast(new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.3, 4, 8), rider));
    shin.position.set(x, 0.37, 0.34); shin.rotation.x = 0.25; group.add(shin);
  }

  return { group, wheelL, wheelR, casters, headMat, ledMat, ledOn: new THREE.Color(PALETTE.rosa), lightLevel: 0, ledFlash: 0 };
}
