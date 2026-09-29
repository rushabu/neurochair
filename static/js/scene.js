// NeuroChair 3D scene: a procedurally-built wheelchair in a night-time smart courtyard.
// Everything is made from Three.js primitives -- no model files to download.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const WORLD_RADIUS = 28;
const CHAIR_RADIUS = 0.45;
const SPEEDS = [0.9, 1.6, 2.4];          // m/s for speed levels 1..3
const TURN_RATE = 2.2;                   // rad/s

const COLORS = {
  cyan: 0x3ef2ff, magenta: 0xff3ea5, amber: 0xffb547, red: 0xff2d4a, green: 0x4dff9a, violet: 0x8a6bff,
};

export function createScene(canvas, { onBlocked } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070f);
  scene.fog = new THREE.FogExp2(0x070a18, 0.035);

  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
  camera.position.set(0, 3, -6);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.minDistance = 2;
  controls.maxDistance = 25;
  controls.enabled = false;

  // ---------- lighting ----------
  scene.add(new THREE.HemisphereLight(0x3a4a8a, 0x0a0a12, 0.35));
  const moon = new THREE.DirectionalLight(0x8fa8ff, 0.55);
  moon.position.set(-12, 20, -8);
  moon.castShadow = true;
  moon.shadow.mapSize.set(2048, 2048);
  Object.assign(moon.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30, far: 60 });
  scene.add(moon);
  // soft fill from the camera side so the chair never turns into a silhouette
  const fill = new THREE.PointLight(0x9fb4ff, 5, 9, 1.5);
  camera.add(fill); fill.position.set(0.6, 0.8, 0.5);
  scene.add(camera);

  // ---------- sky dome + stars ----------
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(90, 32, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: {},
      vertexShader: `varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `varying vec3 vP; void main(){
        float h = clamp(vP.y*1.4, 0.0, 1.0);
        vec3 horizon = vec3(0.07,0.025,0.11); vec3 zenith = vec3(0.005,0.008,0.03);
        gl_FragColor = vec4(mix(horizon, zenith, pow(h,0.6)), 1.0); }`,
    }),
  );
  scene.add(sky);
  {
    const n = 1400, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const th = Math.random() * Math.PI * 2, ph = Math.acos(Math.random() * 0.9 + 0.1);
      pos.set([85 * Math.sin(ph) * Math.cos(th), 85 * Math.cos(ph), 85 * Math.sin(ph) * Math.sin(th)], i * 3);
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffffff, size: 0.35, fog: false, transparent: true, opacity: 0.8 })));
  }

  // ---------- ground ----------
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(60, 64),
    new THREE.MeshStandardMaterial({ color: 0x0b0f1e, roughness: 0.42, metalness: 0.35 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const grid = new THREE.GridHelper(WORLD_RADIUS * 2, WORLD_RADIUS * 2, COLORS.cyan, 0x1a2a55);
  grid.material.transparent = true; grid.material.opacity = 0.18; grid.position.y = 0.002;
  scene.add(grid);

  // glowing walkway cross
  const pathMat = new THREE.MeshStandardMaterial({ color: 0x141a33, roughness: 0.6, metalness: 0.2 });
  const edgeMat = new THREE.MeshBasicMaterial({ color: 0x1c7f8f });
  for (const rot of [0, Math.PI / 2]) {
    const p = new THREE.Mesh(new THREE.PlaneGeometry(3, WORLD_RADIUS * 2), pathMat);
    p.rotation.set(-Math.PI / 2, 0, rot); p.position.y = 0.004; p.receiveShadow = true; scene.add(p);
    for (const side of [-1.5, 1.5]) {
      const e = new THREE.Mesh(new THREE.PlaneGeometry(0.05, WORLD_RADIUS * 2), edgeMat);
      e.rotation.set(-Math.PI / 2, 0, rot);
      if (rot === 0) e.position.set(side, 0.006, 0); else e.position.set(0, 0.006, side);
      scene.add(e);
    }
  }

  // boundary ring
  const boundary = new THREE.Mesh(
    new THREE.TorusGeometry(WORLD_RADIUS + 0.6, 0.05, 8, 160),
    new THREE.MeshBasicMaterial({ color: COLORS.magenta }),
  );
  boundary.rotation.x = Math.PI / 2; boundary.position.y = 0.4; scene.add(boundary);

  // ---------- obstacles ----------
  const obstacles = []; // {x, z, r, kind}
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x1b2140, roughness: 0.3, metalness: 0.7 });
  const addPillar = (x, z, color) => {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.5, 5, 24), pillarMat);
    body.position.y = 2.5; body.castShadow = true; body.receiveShadow = true; g.add(body);
    for (const y of [0.6, 2.4, 4.2]) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.5, 0.03, 8, 40), new THREE.MeshBasicMaterial({ color }));
      ring.rotation.x = Math.PI / 2; ring.position.y = y; g.add(ring);
    }
    g.position.set(x, 0, z); scene.add(g);
    obstacles.push({ x, z, r: 0.5, kind: 'pillar' });
  };
  const addPlanter = (x, z) => {
    const g = new THREE.Group();
    const box = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.6, 1.4), new THREE.MeshStandardMaterial({ color: 0x232a45, roughness: 0.7 }));
    box.position.y = 0.3; box.castShadow = true; box.receiveShadow = true; g.add(box);
    const rim = new THREE.Mesh(new THREE.BoxGeometry(1.45, 0.03, 1.45), new THREE.MeshBasicMaterial({ color: COLORS.green }));
    rim.position.y = 0.61; g.add(rim);
    const bushMat = new THREE.MeshStandardMaterial({ color: 0x1f6b4a, roughness: 0.9, flatShading: true });
    for (let i = 0; i < 3; i++) {
      const b = new THREE.Mesh(new THREE.IcosahedronGeometry(0.45 + Math.random() * 0.2, 0), bushMat);
      b.position.set((Math.random() - 0.5) * 0.6, 0.95 + Math.random() * 0.2, (Math.random() - 0.5) * 0.6);
      b.castShadow = true; g.add(b);
    }
    g.position.set(x, 0, z); scene.add(g);
    obstacles.push({ x, z, r: 0.95, kind: 'planter' });
  };
  const addLamp = (x, z) => {
    const g = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 3.4, 12), pillarMat);
    pole.position.y = 1.7; pole.castShadow = true; g.add(pole);
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.18, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffd9a0 }));
    bulb.position.y = 3.45; g.add(bulb);
    const light = new THREE.PointLight(0xffb870, 6, 9, 1.6); light.position.y = 3.3; g.add(light);
    g.position.set(x, 0, z); scene.add(g);
    obstacles.push({ x, z, r: 0.15, kind: 'lamp' });
  };

  for (const [x, z] of [[-6, -6], [6, -6], [-6, 6], [6, 6], [-14, 0], [14, 0], [0, -14], [0, 14]])
    addPillar(x, z, (x + z) % 4 === 0 ? COLORS.cyan : COLORS.magenta);
  for (const [x, z] of [[-10, -10], [10, 10], [-10, 10], [10, -10], [-18, 8], [18, -8], [8, 18], [-8, -18]])
    addPlanter(x, z);
  for (const [x, z] of [[-2.4, 9], [2.4, -9], [9, 2.4], [-9, -2.4], [-2.4, -20], [2.4, 20]])
    addLamp(x, z);

  // ---------- wheelchair ----------
  const chair = buildWheelchair();
  scene.add(chair.group);

  // shockwave rings + spike particles emitted on each recognised command
  const effects = [];
  const sparkGeo = new THREE.BufferGeometry();
  const SPARKS = 400;
  const sparkPos = new Float32Array(SPARKS * 3), sparkVel = new Float32Array(SPARKS * 3), sparkLife = new Float32Array(SPARKS);
  sparkGeo.setAttribute('position', new THREE.BufferAttribute(sparkPos, 3));
  const sparkMat = new THREE.PointsMaterial({ color: COLORS.cyan, size: 0.045, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false });
  const sparks = new THREE.Points(sparkGeo, sparkMat);
  sparks.frustumCulled = false;
  scene.add(sparks);
  let sparkCursor = 0;

  // ---------- post-processing ----------
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.6, 0.4, 0.55);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  // ---------- motion state ----------
  const s = {
    x: 0, z: -3, heading: 0, targetHeading: 0,
    moving: false, speedLevel: 1, velocity: 0,
    lights: false, cameraMode: 'chase', blockedCooldown: 0,
  };

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    bloom.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  const fwd = () => new THREE.Vector3(Math.sin(s.heading), 0, Math.cos(s.heading));

  function collides(x, z) {
    if (Math.hypot(x, z) > WORLD_RADIUS - CHAIR_RADIUS) return 'boundary';
    for (const o of obstacles) if (Math.hypot(x - o.x, z - o.z) < o.r + CHAIR_RADIUS) return o.kind;
    return null;
  }

  function burst(color, count = 70) {
    const c = new THREE.Color(color);
    // expanding ground ring
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.46, 0.5, 64),
      new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.7, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    ring.rotation.x = -Math.PI / 2; ring.position.set(s.x, 0.02, s.z);
    scene.add(ring); effects.push({ mesh: ring, t: 0 });
    // spikes flying out of the neuromorphic core
    sparkMat.color = c;
    for (let i = 0; i < count; i++) {
      const k = sparkCursor++ % SPARKS;
      sparkPos.set([s.x, 0.4, s.z], k * 3);
      const a = Math.random() * Math.PI * 2, up = 1.5 + Math.random() * 2.5, out = 0.8 + Math.random() * 1.6;
      sparkVel.set([Math.cos(a) * out, up, Math.sin(a) * out], k * 3);
      sparkLife[k] = 1;
    }
    chair.coreFlash = 1;
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
    toggleCamera() {
      s.cameraMode = s.cameraMode === 'chase' ? 'orbit' : 'chase';
      controls.enabled = s.cameraMode === 'orbit';
      return s.cameraMode;
    },
    reset() { Object.assign(s, { x: 0, z: -3, heading: 0, targetHeading: 0, moving: false, velocity: 0 }); },
    speedMps: () => SPEEDS[s.speedLevel],
  };

  // ---------- main loop ----------
  const clock = new THREE.Clock();
  const tmp = new THREE.Vector3();
  function frame() {
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.elapsedTime;

    // heading: smooth rotation toward target
    const dh = s.targetHeading - s.heading;
    const turnStep = Math.sign(dh) * Math.min(Math.abs(dh), TURN_RATE * dt);
    s.heading += turnStep;

    // velocity: ease toward target speed
    const target = s.moving ? SPEEDS[s.speedLevel] : 0;
    s.velocity += (target - s.velocity) * Math.min(1, dt * (s.moving ? 2.5 : 5));
    if (Math.abs(s.velocity) < 0.005) s.velocity = 0;

    if (s.velocity > 0) {
      const f = fwd();
      const nx = s.x + f.x * s.velocity * dt, nz = s.z + f.z * s.velocity * dt;
      // proximity sensor looks a bit ahead so it brakes before touching
      const hit = collides(s.x + f.x * 0.35, s.z + f.z * 0.35) && collides(nx + f.x * 0.3, nz + f.z * 0.3);
      if (hit) {
        s.moving = false; s.velocity = 0;
        if (s.blockedCooldown <= 0) { onBlocked?.(hit); burst(COLORS.red, 40); s.blockedCooldown = 1.2; }
      } else { s.x = nx; s.z = nz; }
    }
    s.blockedCooldown -= dt;

    // place chair
    chair.group.position.set(s.x, 0, s.z);
    chair.group.rotation.y = s.heading;
    chair.group.position.y = s.velocity > 0 ? Math.sin(t * 18) * 0.004 * s.velocity : 0;
    // wheels: roll forward + counter-rotate when turning in place
    const roll = s.velocity * dt / 0.3;
    const spin = (turnStep * 0.33) / 0.3;
    chair.wheelL.rotation.x += roll - spin;
    chair.wheelR.rotation.x += roll + spin;
    for (const c of chair.casters) {
      c.wheel.rotation.x += s.velocity * dt / 0.08;
      c.fork.rotation.y += ((Math.abs(dh) > 0.01 ? Math.sign(dh) * 0.6 : 0) - c.fork.rotation.y) * Math.min(1, dt * 6);
    }

    // lights
    const lf = s.lights ? 1 : 0;
    chair.lightLevel += (lf - chair.lightLevel) * Math.min(1, dt * 10);
    chair.spot.intensity = 22 * chair.lightLevel;
    chair.beam.material.uniforms.uOpacity.value = 0.12 * chair.lightLevel;
    chair.headMat.color.setRGB(0.25 + 1.6 * chair.lightLevel, 0.25 + 1.6 * chair.lightLevel, 0.3 + 1.6 * chair.lightLevel);
    chair.tailMat.color.setRGB(0.3 + 1.4 * chair.lightLevel, 0.03, 0.06);
    // neuromorphic core glow: idle breathing + flash on command
    chair.coreFlash = Math.max(0, chair.coreFlash - dt * 1.8);
    const glow = 0.7 + 0.2 * Math.sin(t * 2.5) + chair.coreFlash * 0.9;
    chair.coreMat.color.setRGB(0.24 * glow, 0.95 * glow, glow);
    chair.underglow.intensity = 0.5 + chair.coreFlash * 2;
    chair.halo.material.opacity = 0.3 + chair.coreFlash * 0.4;

    // effects
    for (let i = effects.length - 1; i >= 0; i--) {
      const e = effects[i]; e.t += dt;
      const k = e.t / 1.1;
      e.mesh.scale.setScalar(1 + k * 9);
      e.mesh.material.opacity = 0.7 * Math.max(0, 1 - k);
      if (k >= 1) { scene.remove(e.mesh); e.mesh.geometry.dispose(); e.mesh.material.dispose(); effects.splice(i, 1); }
    }
    for (let k = 0; k < SPARKS; k++) {
      if (sparkLife[k] <= 0) { sparkPos[k * 3 + 1] = -100; continue; }
      sparkLife[k] -= dt * 0.9;
      sparkVel[k * 3 + 1] -= 3.2 * dt;
      for (let a = 0; a < 3; a++) sparkPos[k * 3 + a] += sparkVel[k * 3 + a] * dt;
      if (sparkPos[k * 3 + 1] < 0.02) { sparkPos[k * 3 + 1] = 0.02; sparkVel[k * 3 + 1] *= -0.3; }
    }
    sparkGeo.attributes.position.needsUpdate = true;
    
    // camera
    const chairPos = tmp.set(s.x, 0.6, s.z);
    if (s.cameraMode === 'chase') {
      const f = fwd();
      const desired = new THREE.Vector3(s.x - f.x * 3.3 + f.z * 0.6, 1.75, s.z - f.z * 3.3 - f.x * 0.6);
      camera.position.lerp(desired, Math.min(1, dt * 3));
      camera.lookAt(s.x + f.x * 1.2, 0.75, s.z + f.z * 1.2);
    } else {
      controls.target.lerp(chairPos, Math.min(1, dt * 4));
      controls.update();
    }

    composer.render();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  return api;
}

function buildWheelchair() {
  const group = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({ color: 0xb8c2d8, roughness: 0.25, metalness: 0.9 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x1a1d26, roughness: 0.55, metalness: 0.3 });
  const fabric = new THREE.MeshStandardMaterial({ color: 0x2b3566, roughness: 0.85 });
  const tire = new THREE.MeshStandardMaterial({ color: 0x0c0c10, roughness: 0.9 });
  const accent = new THREE.MeshBasicMaterial({ color: COLORS.cyan });

  const cast = (m) => { m.castShadow = true; m.receiveShadow = true; return m; };
  const tube = (a, b, r = 0.014, mat = metal) => {
    const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
    const len = va.distanceTo(vb);
    const m = cast(new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 10), mat));
    m.position.copy(va).add(vb).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
    group.add(m); return m;
  };

  // big rear wheels
  const makeWheel = (side) => {
    const w = new THREE.Group();
    const t = cast(new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.028, 14, 48), tire));
    t.rotation.y = Math.PI / 2; w.add(t);
    const rimRing = new THREE.Mesh(new THREE.TorusGeometry(0.272, 0.01, 8, 48), metal);
    rimRing.rotation.y = Math.PI / 2; w.add(rimRing);
    const glowRing = new THREE.Mesh(new THREE.TorusGeometry(0.24, 0.006, 6, 48), accent);
    glowRing.rotation.y = Math.PI / 2; glowRing.position.x = side * 0.012; w.add(glowRing);
    const push = new THREE.Mesh(new THREE.TorusGeometry(0.26, 0.009, 8, 48), metal);
    push.rotation.y = Math.PI / 2; push.position.x = side * 0.045; w.add(push);
    for (let i = 0; i < 14; i++) {
      const a = (i / 14) * Math.PI * 2;
      const sp = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 0.27, 4), metal);
      sp.position.set(0, Math.cos(a) * 0.135, Math.sin(a) * 0.135);
      sp.rotation.x = -a;
      w.add(sp);
    }
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.06, 16), dark);
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
    const cw = cast(new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.02, 10, 24), tire));
    cw.rotation.y = Math.PI / 2; wheel.add(cw);
    wheel.position.set(0, -0.12, -0.03);
    fork.add(wheel);
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.1, 0.02), metal);
    arm.position.set(0.028, -0.08, -0.015); fork.add(arm);
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

  // "neuromorphic core" battery/compute pack under the seat
  const coreMat = new THREE.MeshBasicMaterial({ color: COLORS.cyan });
  const pack = cast(new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.12, 0.24), dark));
  pack.position.set(0, 0.36, -0.02); group.add(pack);
  const strip = new THREE.Mesh(new THREE.BoxGeometry(0.31, 0.015, 0.245), coreMat);
  strip.position.set(0, 0.36, -0.02); group.add(strip);

  // control box on right armrest with glowing screen
  const ctrl = cast(new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.05, 0.1), dark));
  ctrl.position.set(-0.25, 0.725, 0.14); group.add(ctrl);
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(0.06, 0.07), coreMat);
  screen.rotation.x = -Math.PI / 2; screen.position.set(-0.25, 0.752, 0.14); group.add(screen);
  const stick = new THREE.Mesh(new THREE.SphereGeometry(0.018, 12, 8), new THREE.MeshBasicMaterial({ color: COLORS.magenta }));
  stick.position.set(-0.25, 0.79, 0.17); group.add(stick);

  // headlights (on front of armrests) + tail lights
  const headMat = new THREE.MeshBasicMaterial({ color: 0x404050 });
  const tailMat = new THREE.MeshBasicMaterial({ color: 0x400810 });
  for (const x of [0.25, -0.25]) {
    const h = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.02, 16), headMat);
    h.rotation.x = Math.PI / 2; h.position.set(x, 0.68, 0.2); group.add(h);
    const tl = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.02, 0.01), tailMat);
    tl.position.set(x * 0.8, 0.95, -0.26); group.add(tl);
  }

  const spot = new THREE.SpotLight(0xe6f4ff, 0, 16, 0.45, 0.55, 1.2);
  spot.position.set(0, 0.7, 0.25);
  spot.castShadow = true;
  spot.shadow.mapSize.set(1024, 1024);
  spot.target.position.set(0, 0, 5);
  group.add(spot, spot.target);

  // volumetric beam: open cone with alpha fading along its length
  const beamGeo = new THREE.ConeGeometry(1.9, 6, 32, 1, true);
  beamGeo.translate(0, -3, 0);
  beamGeo.rotateX(-Math.PI / 2 + 0.12); // point forward (+z), tilted slightly down
  const beam = new THREE.Mesh(beamGeo, new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uOpacity: { value: 0 } },
    vertexShader: `varying float vD; void main(){ vD = clamp(length(position)/6.0, 0.0, 1.0);
      gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0);} `,
    fragmentShader: `uniform float uOpacity; varying float vD; void main(){ gl_FragColor = vec4(0.85,0.93,1.0, uOpacity*pow(1.0-vD,1.6)); }`,
  }));
  beam.position.set(0, 0.68, 0.22);
  group.add(beam);

  // cyan underglow + ground halo
  const underglow = new THREE.PointLight(COLORS.cyan, 1, 1.8, 2);
  underglow.position.set(0, 0.32, 0); group.add(underglow);
  const halo = new THREE.Mesh(
    new THREE.RingGeometry(0.55, 0.6, 64),
    new THREE.MeshBasicMaterial({ color: COLORS.cyan, transparent: true, opacity: 0.4, blending: THREE.AdditiveBlending, depthWrite: false }),
  );
  halo.rotation.x = -Math.PI / 2; halo.position.y = 0.01; group.add(halo);

  // a simple seated rider silhouette so the chair reads at a distance
  const rider = new THREE.MeshStandardMaterial({ color: 0x6f7fb8, roughness: 0.7 });
  const torso = cast(new THREE.Mesh(new THREE.CapsuleGeometry(0.15, 0.3, 6, 12), rider));
  torso.position.set(0, 0.86, -0.1); torso.rotation.x = -0.08; group.add(torso);
  const head = cast(new THREE.Mesh(new THREE.SphereGeometry(0.1, 20, 14), rider));
  head.position.set(0, 1.2, -0.08); group.add(head);
  for (const x of [0.09, -0.09]) {
    const thigh = cast(new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.3, 4, 8), rider));
    thigh.rotation.x = Math.PI / 2; thigh.position.set(x, 0.6, 0.12); group.add(thigh);
    const shin = cast(new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.3, 4, 8), rider));
    shin.position.set(x, 0.37, 0.34); shin.rotation.x = 0.25; group.add(shin);
  }

  return { group, wheelL, wheelR, casters, spot, beam, headMat, tailMat, coreMat, underglow, halo, lightLevel: 0, coreFlash: 0 };
}
