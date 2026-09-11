/* stage.js — Shofar Brain stage
 * Left: real flybody meshes (decimated) + a shofar, on a dark grid.
 * Right: all 139,255 FlyWire neurons as a point cloud, lit by live spikes from the LIF worker.
 * Training loop: cue -> MECH_JO, response read from MN_PROBOSCIS/MN_HEAD spikes, reward -> MB_DAN_REW.
 */
(function () {
	'use strict';

	var STIM = 0.15, TRIAL_INTERVAL = 3.2, CUE_DURATION = 2.0, EFFORT_REF = 6.0, SKILL_GAIN = 0.4;
	function untrainedRef() { return (trained.meta && trained.meta.evoked_base) || 3.6; }
	function skillOf(ev) { return Math.max(0, Math.min(1, (ev / untrainedRef() - 1) / SKILL_GAIN)); }
	var S = { active: false, motorSpikes: 0, fireStates: 0, phaseMotor: 0, phaseTicks: 0, attempt: 0, skill: 0, phase: 'idle', phaseT: 0, effortSum: 0, effortN: 0, lastEffort: 0, baseline: 0, baseSum: 0, baseN: 0, evoked: 0, evokedEMA: 0, plasticOn: false,
		blast: null, performing: false, perfIdx: 0, celebrated: false, ready: false };
	var BLASTS = {
		sputter:  { he: 'Sputter', en: 'no clean note yet', notes: [[0, 0.35]] },
		tekiah:   { he: 'Tekiah', en: 'one long blast', notes: [[0, 1.6]] },
		shevarim: { he: 'Shevarim', en: 'three broken notes', notes: [[0, 0.38], [0.44, 0.38], [0.88, 0.38]] },
		teruah:   { he: 'Teruah', en: 'nine staccato notes', notes: [] },
		gedolah:  { he: 'Tekiah Gedolah', en: 'the great blast', notes: [[0, 4.0]] }
	};
	for (var i = 0; i < 9; i++) BLASTS.teruah.notes.push([i * 0.14, 0.1]);
	// the minimal Rosh Hashanah set: TaShRaT, TaShaT, TaRaT; the closing tekiah is extended (tekiah gedolah)
	var SEQUENCE = ['tekiah', 'shevarim', 'teruah', 'tekiah', 'tekiah', 'shevarim', 'tekiah', 'tekiah', 'teruah', 'gedolah'];
	var $ = function (id) { return document.getElementById(id); };
	BRAIN.setup();                       // creates BRAIN.postSynaptic; main.js normally does this
	BRAIN._isMoving = false;

	/* ---------------- audio (shofar synth) ---------------- */
	var actx = null, master, noiseBuf, wave;
	function audio() {
		if (actx) { if (actx.state === 'suspended') actx.resume(); return actx; }
		actx = new (window.AudioContext || window.webkitAudioContext)();
		master = actx.createGain(); master.gain.value = 0.7; master.connect(actx.destination);
		var real = new Float32Array(12), imag = new Float32Array(12);
		var amps = [0, 1, 0.75, 0.55, 0.42, 0.3, 0.22, 0.15, 0.1, 0.07, 0.05, 0.03];
		for (var k = 1; k < 12; k++) imag[k] = amps[k];
		wave = actx.createPeriodicWave(real, imag);
		var len = actx.sampleRate * 2, b = actx.createBuffer(1, len, actx.sampleRate), d = b.getChannelData(0);
		for (var n = 0; n < len; n++) d[n] = Math.random() * 2 - 1;
		noiseBuf = b; return actx;
	}
	function note(t0, dur, q) {
		var c = audio(), f0 = 392 * (1 + (Math.random() - 0.5) * 0.02 * (1 - q));
		var osc = c.createOscillator(); osc.setPeriodicWave(wave);
		var lfo = c.createOscillator(), lfoG = c.createGain();
		lfo.frequency.value = 5.2 + (1 - q) * 3; lfoG.gain.value = f0 * (0.004 + (1 - q) * 0.03);
		lfo.connect(lfoG); lfoG.connect(osc.frequency);
		osc.frequency.setValueAtTime(f0 * (q < 0.35 ? 0.94 : 1), t0);
		if (q > 0.5) osc.frequency.exponentialRampToValueAtTime(f0 * 1.012, t0 + dur);
		if (q < 0.55 && dur > 0.25) { var tc = t0 + dur * (0.3 + Math.random() * 0.4); osc.frequency.setValueAtTime(f0 * 1.5, tc); osc.frequency.setValueAtTime(f0, tc + 0.07); }
		var lp = c.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 1400 + q * 2600; lp.Q.value = 1.2;
		var g = c.createGain(), peak = 0.35 + 0.45 * q, atk = 0.03 + (1 - q) * 0.08, rel = 0.12;
		g.gain.setValueAtTime(0.0001, t0); g.gain.exponentialRampToValueAtTime(peak, t0 + atk);
		g.gain.setValueAtTime(peak, t0 + Math.max(atk, dur - rel)); g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
		osc.connect(lp); lp.connect(g); g.connect(master);
		var ns = c.createBufferSource(); ns.buffer = noiseBuf; ns.loop = true;
		var nf = c.createBiquadFilter(); nf.type = 'bandpass'; nf.frequency.value = 900; nf.Q.value = 0.7;
		var ng = c.createGain(); ng.gain.setValueAtTime(0.0001, t0);
		ng.gain.exponentialRampToValueAtTime(0.03 + (1 - q) * 0.25, t0 + atk); ng.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
		ns.connect(nf); nf.connect(ng); ng.connect(master);
		osc.start(t0); lfo.start(t0); ns.start(t0); osc.stop(t0 + dur + 0.05); lfo.stop(t0 + dur + 0.05); ns.stop(t0 + dur + 0.05);
	}
	function playBlast(type, q) {
		var c = audio(), b = BLASTS[type], t0 = c.currentTime + 0.02, total = 0, notes = b.notes;
		if (type === 'sputter') notes = [[0, 0.2 + q * 0.4]];
		for (var i = 0; i < notes.length; i++) { note(t0 + notes[i][0], notes[i][1] * (type === 'sputter' ? 1 : 0.8 + 0.2 * q), q); total = Math.max(total, notes[i][0] + notes[i][1]); }
		S.blast = { type: type, t0: performance.now() / 1000, dur: total, q: q, notes: notes };
		$('blast').innerHTML = b.he + '<span>' + b.en + '</span>'; $('blast').classList.add('on');
	}
	function noteActive(now) {
		if (!S.blast) return 0; var el = now - S.blast.t0; if (el > S.blast.dur) return 0;
		for (var i = 0; i < S.blast.notes.length; i++) { var n = S.blast.notes[i]; if (el >= n[0] && el <= n[0] + n[1]) return 1; } return 0;
	}

	/* ---------------- brain I/O ---------------- */
	function readPS(n) { var ps = BRAIN.postSynaptic && BRAIN.postSynaptic[n]; return ps ? (ps[BRAIN.nextState] || 0) : 0; }
	function effortNow() { return readPS('MN_PROBOSCIS') + readPS('MN_HEAD') * 0.6 + readPS('SEZ_FEED') * 0.4; }
	function setCue(on) {
		BRAIN.stimulate.extraSegments = on ? [{ name: 'MECH_JO', intensity: STIM * 3 }] : null;
		$('cueState').textContent = on ? '♪ sound cue → ear neurons' : '';
		$('cueState').style.color = on ? '#e0b04a' : '';
	}
	function reward(str) {
		// dopamine as a modulatory third factor (same as the offline trainer): raises the worker's DA trace, injects no spikes
		if (BRAIN.worker) BRAIN.worker.postMessage({ type: 'dopamine', amount: ((trained.meta && trained.meta.da_signal) || 1) * str });
		flashGroup('MB_DAN_REW');
		if (BRAIN.drives) BRAIN.drives.fear = Math.max(0, (BRAIN.drives.fear || 0) - 0.3);
	}
	var trained = { meta: null, edges: null, w: null, on: false, loaded: false, fetched: false };
	function applyRegime() { // same network regime the offline training used
		var m = trained.meta || { gain: 150, cap: 0.6, adapt_inc: 1.0, adapt_decay: 0.97, no_tonic: true };
		BRAIN.worker.postMessage({ type: 'setWeightTransform', gain: m.gain, cap: m.cap });
		BRAIN.worker.postMessage({ type: 'setParams', adaptInc: m.adapt_inc || 0, adaptDecay: m.adapt_decay || 0.97 });
		BRAIN.stimulate.noTonic = !!m.no_tonic; BRAIN.stimulate.labMode = true;
		$('regime').textContent = 'gain ' + m.gain + ' · cap ' + m.cap + ' · adaptation ' + (m.adapt_inc || 0) + '/' + (m.adapt_decay || 0.97) + (m.no_tonic ? ' · silent background' : '');
	}
	function setTrained(on) {
		if (!BRAIN.worker) return;
		applyRegime();
		if (on && trained.edges) BRAIN.worker.postMessage({ type: 'setWeights', edges: trained.edges, w: trained.w });
		trained.on = on && !!trained.edges;
		// skill shown before any trial = the trainer's measured test response of this brain
		if (!S.active && !S.performing) { S.evokedEMA = trained.on && trained.meta ? (trained.meta.evoked_test || 0) : (trained.meta ? trained.meta.evoked_base : 0); S.skill = skillOf(S.evokedEMA); }
		$('brainState').textContent = trained.on ? 'trained · ' + trained.edges.length.toLocaleString() + ' synapses learned by dopamine reward' : 'untrained · original FlyWire weights';
		$('brainState').style.color = trained.on ? '#e0b04a' : '';
	}
	fetch('stage/trained_meta.json').then(function (r) { return r.ok ? r.json() : null; }).then(function (m) { trained.meta = m; return fetch('stage/trained_weights.bin'); })
		.then(function (r) { return r.ok ? r.arrayBuffer() : null; }).then(function (buf) {
			if (!buf) return;
			var n = new DataView(buf).getUint32(0, true);
			trained.edges = new Uint32Array(buf.slice(4, 4 + n * 4)); trained.w = new Float32Array(buf.slice(4 + n * 4, 4 + n * 8)); trained.loaded = true;
			if (S.plasticOn) setTrained(true);
		}).catch(function (e) { console.warn('no trained weights', e); }).then(function () { trained.fetched = true; });
	function enablePlasticity() {
		var n2i = BRAIN.workerGroupNameToId; if (!BRAIN.worker || !n2i || S.plasticOn || !trained.fetched) return;
		applyRegime();
		var ids = function (names) { return names.map(function (n) { return n2i[n]; }).filter(function (x) { return x !== undefined; }); };
		var m = trained.meta || {};
		// plastic set = the offline trainer's edge list when available (cue-path presynaptic -> mouth motor); dopamine arrives as 'dopamine' messages, not DAN spikes
		BRAIN.worker.postMessage({ type: 'plasticity', postGroups: ids(m.plastic_post || ['SEZ_FEED', 'MN_PROBOSCIS', 'MN_HEAD']), daGroups: [], edges: trained.edges || undefined, eta: m.eta || 0.002 });
		S.plasticOn = true;
		if (trained.loaded) setTrained(true); else $('brainState').textContent = 'UNTRAINED brain · original FlyWire weights';
	}
	function stageFor(s) { return s < 0.3 ? 'sputter' : s < 0.55 ? 'tekiah' : s < 0.75 ? 'shevarim' : s < 0.93 ? 'teruah' : 'gedolah'; }

	function step(dt) {
		if (!S.active) return;
		S.phaseT += dt;
		if (S.phase === 'idle') {
			var left = S.blast ? (S.blast.t0 + S.blast.dur + 0.25) - performance.now() / 1000 : 0;
			if (S.phaseT >= TRIAL_INTERVAL - CUE_DURATION && left <= 0 && !(S.performing && S.perfIdx >= SEQUENCE.length)) {
				S.baseline = S.phaseTicks ? S.phaseMotor / S.phaseTicks : 0; S.phaseMotor = 0; S.phaseTicks = 0;
				S.attempt++; S.phase = 'cue'; S.phaseT = 0; setCue(true);
			}
		} else if (S.phase === 'cue') {
			if (S.phaseT >= CUE_DURATION) {
				setCue(false);
				var effort = S.phaseTicks ? S.phaseMotor / S.phaseTicks : 0; S.lastEffort = effort; S.phaseMotor = 0; S.phaseTicks = 0;
				// evoked = cue-driven motor activity above the pre-cue baseline (measured, not assigned)
				S.evoked = Math.max(0, effort - S.baseline);
				S.evokedEMA = S.evokedEMA * 0.65 + S.evoked * 0.35;
				var effNorm = skillOf(S.evoked);
				if (S.performing) {
					// performance = test trials: no dopamine; each blast's quality is the measured cue response of the current brain
					var want = SEQUENCE[S.perfIdx++];
					S.skill = skillOf(S.evokedEMA);
					playBlast(effNorm >= 0.3 ? want : 'sputter', Math.min(1, 0.12 + effNorm * 0.95));
				} else {
					// reinforcement: dopamine when the cue produced a mouth response (the offline trainer's 'reward' arm)
					if (S.evoked > 0.5) reward(Math.min(1.5, 0.3 + S.evoked / 4));
					S.skill = skillOf(S.evokedEMA);
					playBlast(stageFor(S.skill), Math.min(1, 0.12 + effNorm * 0.95));
				}
				S.phase = 'idle'; S.phaseT = 0;
				// like the trainer: hard reset of membrane/adaptation state between trials (weights and learning persist)
				if (BRAIN.worker) BRAIN.worker.postMessage({ type: 'reset' });
				if (!S.performing && S.skill >= 0.93 && !S.celebrated) { S.celebrated = true; perform(); }
			}
		}
	}
	function stepPerformance(dt) {
		if (!S.performing || S.perfIdx < SEQUENCE.length || S.phase !== 'idle') return;
		var left = S.blast ? (S.blast.t0 + S.blast.dur + 0.4) - performance.now() / 1000 : 0;
		if (left > 0) return;
		S.performing = false; S.active = false; setCue(false); $('btnTrain').classList.remove('on');
		$('blast').innerHTML = 'Shana Tova<span>a happy new year · 5787</span>'; $('blast').classList.add('on');
	}
	function hud() {
		var pct = Math.round(S.skill * 100);
		$('tAttempt').textContent = FT.on ? 'trial ' + FT.trial + ' of ' + FT.trials : S.performing ? 'blast ' + Math.min(SEQUENCE.length, S.perfIdx + 1) + ' of ' + SEQUENCE.length : S.attempt ? 'attempt ' + S.attempt : 'trained brain';
		$('tSkill').textContent = pct + '%'; $('tSkillBar').style.width = pct + '%';
		$('tEffort').style.width = Math.min(100, Math.round(S.lastEffort / (untrainedRef() * (1 + SKILL_GAIN)) * 100)) + '%';
		var idle = !S.active && !S.performing && !FT.on && S.attempt === 0;
		$('stageHe').textContent = FT.on ? 'Learning' : S.performing ? 'Shofar service' : idle ? 'Ready' : BLASTS[stageFor(S.skill)].he;
		$('stageEn').textContent = FT.on ? 'dopamine reward when the mouth answers the sound · fast-forward' : S.performing ? BLASTS[SEQUENCE[Math.min(SEQUENCE.length - 1, S.perfIdx)]].he + ' · ' + BLASTS[SEQUENCE[Math.min(SEQUENCE.length - 1, S.perfIdx)]].en : idle ? 'press B · the fly blows the shofar service' : 'next: ' + BLASTS[stageFor(S.skill)].en;
		var plf = BRAIN.plasticity;
		$('tPhase').textContent = FT.on ? 'mouth response ' + S.evoked.toFixed(1) + ' spikes/tick · synapses Δw +' + (plf && plf.sumW0 ? (plf.sumDelta / plf.sumW0 * 100).toFixed(0) : '0') + '%' : S.phase === 'cue' ? 'sound cue on · reading mouth motor neurons' : (S.active || S.performing) ? 'mouth response ' + S.evoked.toFixed(1) + ' spikes/tick · untrained ' + untrainedRef().toFixed(1) + (S.performing ? ' · no reward' : '') : '';
		var pl = BRAIN.plasticity;
		$('tSyn').textContent = pl ? ('Δw +' + (pl.sumW0 ? (pl.sumDelta / pl.sumW0 * 100) : 0).toFixed(2) + '% on ' + pl.edges.toLocaleString() + ' synapses · DA ' + pl.da.toFixed(2)) : (S.plasticOn ? 'building plastic edge list…' : 'plasticity off');
		if (S.blast && performance.now() / 1000 - S.blast.t0 > S.blast.dur + 0.5 && !(S.performing === false && S.perfIdx >= SEQUENCE.length && S.celebrated)) $('blast').classList.remove('on');
	}

	/* ---------------- fly scene ---------------- */
	var flyCanvas = $('flyCanvas'), fr = new THREE.WebGLRenderer({ canvas: flyCanvas, antialias: true, alpha: true });
	fr.setPixelRatio(Math.min(2, window.devicePixelRatio)); fr.outputEncoding = THREE.sRGBEncoding; fr.toneMapping = THREE.ACESFilmicToneMapping;
	var fs = new THREE.Scene(); fs.fog = new THREE.FogExp2(0x05070d, 0.09);
	var fcam = new THREE.PerspectiveCamera(38, 1, 0.05, 100); fcam.position.set(4.6, 2.3, 4.9);
	var fctl = new THREE.OrbitControls(fcam, flyCanvas); fctl.target.set(0, 0.35, 0); fctl.autoRotate = true; fctl.autoRotateSpeed = 0.45; fctl.enableDamping = true; fctl.minDistance = 1.5; fctl.maxDistance = 9;
	fs.add(new THREE.HemisphereLight(0x8fb7ff, 0x1a1208, 0.55));
	var key = new THREE.DirectionalLight(0xfff1d6, 1.4); key.position.set(3, 5, 2); fs.add(key);
	var rim = new THREE.DirectionalLight(0x6fd3ff, 0.6); rim.position.set(-4, 2, -3); fs.add(rim);
	var bellLight = new THREE.PointLight(0xe0b04a, 0, 4, 2); fs.add(bellLight);
	var grid = new THREE.GridHelper(30, 60, 0x24304a, 0x131a2a); grid.position.y = 0; fs.add(grid);
	var floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: 0x080b14, roughness: 1 }));
	floor.rotation.x = -Math.PI / 2; floor.position.y = -0.002; fs.add(floor);
	var flyG = new THREE.Group(); fs.add(flyG);
	var mouthWorld = new THREE.Vector3(0.9, 0.35, 0), fwdWorld = new THREE.Vector3(1, 0, 0);
	var horn = null, hornMat = null, bellWorld = new THREE.Vector3(), bellDir = new THREE.Vector3();

	fetch('stage/fly.bin').then(function (r) { return r.arrayBuffer(); }).then(function (buf) {
		var dv = new DataView(buf), nV = dv.getUint32(0, true), nI = dv.getUint32(4, true), o = 8;
		var pos = new Float32Array(buf, o, nV * 3); o += nV * 12;
		var nrm = new Float32Array(buf, o, nV * 3); o += nV * 12;
		var col = new Uint8Array(buf, o, nV * 4); o += nV * 4;
		var idx = new Uint32Array(buf, o, nI);
		// orientation: longest extent = body axis, shortest = up; eyes (red) mark the front
		var mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9], eye = [0, 0, 0], ne = 0, ctr = [0, 0, 0];
		for (var v = 0; v < nV; v++) for (var a = 0; a < 3; a++) { var p = pos[v * 3 + a]; if (p < mn[a]) mn[a] = p; if (p > mx[a]) mx[a] = p; ctr[a] += p / nV; }
		for (var v2 = 0; v2 < nV; v2++) if (col[v2 * 4] > 190 && col[v2 * 4 + 1] < 80) { ne++; for (var a2 = 0; a2 < 3; a2++) eye[a2] += pos[v2 * 3 + a2]; }
		if (ne) for (var a3 = 0; a3 < 3; a3++) eye[a3] /= ne;
		var ext = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]], off = [eye[0] - ctr[0], eye[1] - ctr[1], eye[2] - ctr[2]];
		var fa = off.map(Math.abs).indexOf(Math.max.apply(null, off.map(Math.abs)));           // eyes sit at the front of the head
		var others = [0, 1, 2].filter(function (a) { return a !== fa; });
		var ua = ext[others[0]] < ext[others[1]] ? others[0] : others[1];                       // the wings span the widest remaining axis; up is the thinnest
		var F = new THREE.Vector3(); F.setComponent(fa, Math.sign(eye[fa] - ctr[fa]) || 1);
		var U = new THREE.Vector3(); U.setComponent(ua, 1);
		var R = new THREE.Vector3().crossVectors(F, U).normalize();  // right-handed: x=forward, y=up, z=right
		var basis = new THREE.Matrix4().makeBasis(F, U, R); // maps local F->x, U->y, R->z
		var rot = new THREE.Matrix4().copy(basis).transpose();
		var scale = 2.6 / ext[fa];
		var geo = new THREE.BufferGeometry();
		geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
		geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
		var cf = new Float32Array(nV * 3); for (var c = 0; c < nV; c++) { cf[c * 3] = col[c * 4] / 255; cf[c * 3 + 1] = col[c * 4 + 1] / 255; cf[c * 3 + 2] = col[c * 4 + 2] / 255; }
		geo.setAttribute('color', new THREE.BufferAttribute(cf, 3));
		var opaque = [], trans = [];
		for (var t = 0; t < nI; t += 3) { (col[idx[t] * 4 + 3] < 255 ? trans : opaque).push(idx[t], idx[t + 1], idx[t + 2]); }
		var gO = geo.clone(); gO.setIndex(new THREE.BufferAttribute(new Uint32Array(opaque), 1));
		var gT = geo.clone(); gT.setIndex(new THREE.BufferAttribute(new Uint32Array(trans), 1));
		var mO = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.08 });
		var mT = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.2, metalness: 0.3, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false });
		var body = new THREE.Group(); body.add(new THREE.Mesh(gO, mO)); body.add(new THREE.Mesh(gT, mT));
		body.applyMatrix4(new THREE.Matrix4().makeTranslation(-ctr[0], -ctr[1], -ctr[2]));
		var wrap = new THREE.Group(); wrap.add(body); wrap.applyMatrix4(rot); wrap.scale.setScalar(scale);
		flyG.add(wrap);
		// ground it
		var bb = new THREE.Box3().setFromObject(flyG); flyG.position.y -= bb.min.y; flyG.position.y += 0.02;
		var eyeL = new THREE.Vector3(eye[0] - ctr[0], eye[1] - ctr[1], eye[2] - ctr[2]).applyMatrix4(rot).multiplyScalar(scale);
		mouthWorld.copy(eyeL).add(flyG.position).add(new THREE.Vector3(0.12, -0.22, 0));  // proboscis: just ahead of and below the eyes
		fwdWorld.set(1, 0, 0);
		buildHorn();
		fctl.target.set(0.8, 0.95, 0);
	}).catch(function (e) { console.error('fly.bin', e); buildHorn(); });

	function buildHorn() {
		var m = mouthWorld, f = fwdWorld;
		var pts = [m.clone(), m.clone().addScaledVector(f, 0.35).add(new THREE.Vector3(0, 0.05, 0.04)), m.clone().addScaledVector(f, 0.75).add(new THREE.Vector3(0, 0.32, 0.12)), m.clone().addScaledVector(f, 1.05).add(new THREE.Vector3(0, 0.78, 0.22)), m.clone().addScaledVector(f, 1.15).add(new THREE.Vector3(0, 1.2, 0.34))];
		var curve = new THREE.CatmullRomCurve3(pts), N = 48, R = 14, verts = [], norms = [], uvs = [], idx = [];
		for (var i = 0; i <= N; i++) {
			var t = i / N, p = curve.getPointAt(t), tan = curve.getTangentAt(t), r = 0.028 + Math.pow(t, 1.8) * 0.2;
			var nrmA = new THREE.Vector3(0, 1, 0).cross(tan).normalize(), bin = tan.clone().cross(nrmA).normalize();
			for (var j = 0; j <= R; j++) {
				var ang = j / R * Math.PI * 2, nx = nrmA.clone().multiplyScalar(Math.cos(ang)).addScaledVector(bin, Math.sin(ang));
				verts.push(p.x + nx.x * r, p.y + nx.y * r, p.z + nx.z * r); norms.push(nx.x, nx.y, nx.z); uvs.push(t, j / R);
			}
			if (i === N) { bellWorld.copy(p); bellDir.copy(tan); }
		}
		for (var a = 0; a < N; a++) for (var b = 0; b < R; b++) { var k = a * (R + 1) + b; idx.push(k, k + R + 1, k + 1, k + 1, k + R + 1, k + R + 2); }
		var g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(norms, 3)); g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2)); g.setIndex(idx);
		// horn texture: ivory→dark ram's-horn bands
		var cv = document.createElement('canvas'); cv.width = 256; cv.height = 8; var cx = cv.getContext('2d');
		var gr = cx.createLinearGradient(0, 0, 256, 0); gr.addColorStop(0, '#3b2a14'); gr.addColorStop(0.35, '#8a6a3a'); gr.addColorStop(0.7, '#d9c08c'); gr.addColorStop(1, '#f1e6cc'); cx.fillStyle = gr; cx.fillRect(0, 0, 256, 8);
		for (var s = 0; s < 22; s++) { cx.fillStyle = 'rgba(40,25,10,' + (0.25 + Math.random() * 0.3) + ')'; cx.fillRect(20 + s * 10.5 + Math.random() * 3, 0, 1.5 + Math.random() * 2, 8); }
		var tex = new THREE.CanvasTexture(cv); tex.encoding = THREE.sRGBEncoding;
		hornMat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.35, metalness: 0.15, emissive: 0xe0b04a, emissiveIntensity: 0 });
		horn = new THREE.Mesh(g, hornMat); fs.add(horn);
		bellLight.position.copy(bellWorld);
	}

	// blast FX: rings + embers
	var rings = [], ringGeo = new THREE.RingGeometry(0.16, 0.19, 48), ringMat = new THREE.MeshBasicMaterial({ color: 0xf3d489, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending });
	var PN = 400, pPos = new Float32Array(PN * 3), pVel = new Float32Array(PN * 3), pLife = new Float32Array(PN), pHead = 0;
	var pGeo = new THREE.BufferGeometry(); pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
	var pMat = new THREE.PointsMaterial({ color: 0xe0b04a, size: 0.035, transparent: true, opacity: 0.9, depthWrite: false, blending: THREE.AdditiveBlending });
	var pts = new THREE.Points(pGeo, pMat); pts.frustumCulled = false; fs.add(pts);
	function fx(dt, on, q) {
		if (horn) { hornMat.emissiveIntensity += ((on ? 0.35 + q * 0.5 : 0) - hornMat.emissiveIntensity) * 0.25; bellLight.intensity += ((on ? 2.2 + q * 2 : 0) - bellLight.intensity) * 0.25; }
		if (on && Math.random() < 0.55 + q * 0.4) { var r = new THREE.Mesh(ringGeo, ringMat.clone()); r.position.copy(bellWorld); r.lookAt(bellWorld.clone().add(bellDir)); r.userData.life = 1; rings.push(r); fs.add(r); }
		for (var i = rings.length - 1; i >= 0; i--) { var rg = rings[i]; rg.userData.life -= dt * 1.4; rg.position.addScaledVector(bellDir, dt * 1.6); var sc = 1 + (1 - rg.userData.life) * 4.5; rg.scale.set(sc, sc, sc); rg.material.opacity = rg.userData.life * 0.6 * (0.4 + q); if (rg.userData.life <= 0) { fs.remove(rg); rings.splice(i, 1); } }
		if (on) for (var k = 0; k < 6; k++) { var h = pHead = (pHead + 1) % PN; pPos[h * 3] = bellWorld.x; pPos[h * 3 + 1] = bellWorld.y; pPos[h * 3 + 2] = bellWorld.z; pVel[h * 3] = bellDir.x * 1.4 + (Math.random() - 0.5) * 0.8; pVel[h * 3 + 1] = bellDir.y * 1.4 + Math.random() * 0.9; pVel[h * 3 + 2] = bellDir.z * 1.4 + (Math.random() - 0.5) * 0.8; pLife[h] = 1; }
		for (var p = 0; p < PN; p++) { if (pLife[p] <= 0) { pPos[p * 3 + 1] = -99; continue; } pLife[p] -= dt * 0.9; pVel[p * 3 + 1] -= dt * 0.6; pPos[p * 3] += pVel[p * 3] * dt; pPos[p * 3 + 1] += pVel[p * 3 + 1] * dt; pPos[p * 3 + 2] += pVel[p * 3 + 2] * dt; }
		pGeo.attributes.position.needsUpdate = true;
	}

	/* ---------------- brain scene ---------------- */
	var brainCanvas = $('brainCanvas'), br = new THREE.WebGLRenderer({ canvas: brainCanvas, antialias: false, alpha: true });
	br.setPixelRatio(Math.min(2, window.devicePixelRatio));
	var bs = new THREE.Scene(), bcam = new THREE.PerspectiveCamera(40, 1, 0.01, 50); bcam.position.set(0, 0.25, 2.9);
	var bctl = new THREE.OrbitControls(bcam, brainCanvas); bctl.autoRotate = false; bctl.enableDamping = true; bctl.enablePan = false; bctl.target.set(0, 0.05, 0);
	var N = 0, act = null, cloud = null, cat = null; brainCanvas.addEventListener('pointerdown', function () { bctl._userMoved = true; });
	fetch('stage/brain_points.bin').then(function (r) { return r.arrayBuffer(); }).then(function (buf) {
		var dv = new DataView(buf); N = dv.getUint32(0, true);
		var P = new Float32Array(buf, 4, N * 3); cat = new Uint8Array(buf, 4 + N * 12, N);
		act = new Float32Array(N); var catF = new Float32Array(N); for (var i = 0; i < N; i++) catF[i] = cat[i];
		var g = new THREE.BufferGeometry();
		g.setAttribute('position', new THREE.BufferAttribute(P, 3)); g.setAttribute('cat', new THREE.BufferAttribute(catF, 1)); g.setAttribute('act', new THREE.BufferAttribute(act, 1));
		var mat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
			uniforms: { uPR: { value: Math.min(2, window.devicePixelRatio) } },
			vertexShader: 'attribute float cat; attribute float act; varying float vA; varying float vC; uniform float uPR;' +
				'void main(){ vA=act; vC=cat; vec4 mv=modelViewMatrix*vec4(position,1.0); gl_Position=projectionMatrix*mv; gl_PointSize=(2.3+act*3.6)*uPR*(2.2/-mv.z); }',
			fragmentShader: 'varying float vA; varying float vC;' +
				'void main(){ vec2 d=gl_PointCoord-0.5; float r=dot(d,d); if(r>0.25) discard; float soft=1.0-smoothstep(0.05,0.25,r);' +
				'vec3 central=vec3(0.95,0.72,0.34); vec3 optic=vec3(0.38,0.74,1.0); vec3 base=mix(central,optic,vC);' +
				'vec3 col=mix(base*0.8, vec3(1.0,0.97,0.88), clamp(vA*1.1,0.0,1.0)); float a=(0.3+vA*0.9)*soft; gl_FragColor=vec4(col*a,a); }'
		});
		cloud = new THREE.Points(g, mat); cloud.rotation.x = Math.PI; cloud.frustumCulled = false; bs.add(cloud);
		$('nCount').textContent = N.toLocaleString() + ' neurons';
	}).catch(function (e) { console.error('brain_points.bin', e); $('nCount').textContent = 'point cloud failed: ' + e.message; });

	/* spikes/s + raster */
	var lastFire = null, spikeWin = [], rasterIdx = null, rc = $('raster'), rctx = rc.getContext('2d'), rasterX = 0;
	function pickRaster() {
		var arr = BRAIN.workerGroupIdArr; if (!arr) return;
		rasterIdx = new Uint32Array(96); var n = 0, tries = 0;
		while (n < 96 && tries++ < 200000) { var i = Math.floor(Math.random() * arr.length); if (cat && cat[i] === 1 && Math.random() < 0.7) continue; rasterIdx[n++] = i; }
	}
	function tickSpikes(now) {
		var f = BRAIN.latestFireState; if (!f || f === lastFire) return; lastFire = f;
		var sum = 0, garr = BRAIN.workerGroupIdArr; for (var i = 0; i < f.length; i++) { var s = f[i]; if (s) { sum++; if (act) act[i] = 1; } }
		S.fireStates++;
		if (act) for (var j = 0; j < f.length; j++) act[j] *= 0.86;
		if (cloud) cloud.geometry.attributes.act.needsUpdate = true;
		spikeWin.push([now, sum]); while (spikeWin.length && now - spikeWin[0][0] > 1) spikeWin.shift();
		var tot = 0; for (var k = 0; k < spikeWin.length; k++) tot += spikeWin[k][1];
		$('spikeRate').textContent = tot >= 1e6 ? (tot / 1e6).toFixed(2) + 'M' : tot >= 1e3 ? (tot / 1e3).toFixed(1) + 'K' : String(tot);
		// raster column
		if (!rasterIdx) pickRaster();
		if (rasterIdx) {
			var W = rc.width, H = rc.height; if (rasterX >= W) { rctx.drawImage(rc, -2, 0); rasterX = W - 2; }
			rctx.fillStyle = 'rgba(5,7,13,0.9)'; rctx.fillRect(rasterX, 0, 2, H);
			for (var r = 0; r < 96; r++) if (f[rasterIdx[r]]) { rctx.fillStyle = cat && cat[rasterIdx[r]] ? '#6fd3ff' : '#e0b04a'; rctx.fillRect(rasterX, (r / 96) * H, 1.5, Math.max(1, H / 96)); }
			rasterX += 2;
		}
	}
	function flashGroup(name) { var n2i = BRAIN.workerGroupNameToId, garr = BRAIN.workerGroupIdArr; if (!act || !n2i || !garr || n2i[name] === undefined) return; var g = n2i[name]; for (var i = 0; i < garr.length; i++) if (garr[i] === g) act[i] = 1; if (cloud) cloud.geometry.attributes.act.needsUpdate = true; }
	var motorGid = null;
	BRAIN.onWorkerTick = function (d) {
		var gsc = d.groupSpikeCounts; if (!gsc || !motorGid) return;
		var m = 0; for (var g in motorGid) m += gsc[g] || 0;
		S.motorSpikes += m; S.phaseMotor += m; S.phaseTicks++;
	};
	function initMotorGid() { var n2i = BRAIN.workerGroupNameToId; if (!n2i || motorGid) return; motorGid = {}; ['MN_PROBOSCIS', 'MN_HEAD', 'SEZ_FEED'].forEach(function (n) { if (n2i[n] !== undefined) motorGid[n2i[n]] = true; }); }
	function sizeRaster() { var dpr = Math.min(2, window.devicePixelRatio); rc.width = rc.clientWidth * dpr; rc.height = rc.clientHeight * dpr; rctx.fillStyle = '#05070d'; rctx.fillRect(0, 0, rc.width, rc.height); rasterX = 0; }

	/* ---------------- loop ---------------- */
	function resize() {
		var a = flyCanvas.parentElement.getBoundingClientRect(), b = brainCanvas.parentElement.getBoundingClientRect();
		fr.setSize(a.width, a.height, false); fcam.aspect = a.width / a.height; fcam.updateProjectionMatrix();
		br.setSize(b.width, b.height, false); bcam.aspect = b.width / b.height; bcam.updateProjectionMatrix(); sizeRaster();
	}
	window.addEventListener('resize', resize); resize();
	var last = performance.now(), readyShown = false;
	function loop(ts) {
		requestAnimationFrame(loop);
		var dt = Math.min(0.1, (ts - last) / 1000); last = ts; var now = ts / 1000;
		if (!S.ready && BRAIN.workerGroupIdArr) { S.ready = true; }
		if (S.ready && !readyShown) { readyShown = true; $('loading').classList.add('hide'); }
		if (S.ready && BRAIN.update) { BRAIN._isMoving = false; BRAIN.update(); enablePlasticity(); initMotorGid(); }
		step(dt); stepPerformance(dt); hud();
		var on = noteActive(now), q = S.blast ? S.blast.q : 0;
		fx(dt, on, q); tickSpikes(now);
		if (horn) { horn.visible = true; }
		if (!bctl._userMoved) { var sw = Math.sin(now * 0.35) * 0.55; bcam.position.set(Math.sin(sw) * 2.9, 0.25 + Math.sin(now * 0.2) * 0.1, Math.cos(sw) * 2.9); bcam.lookAt(bctl.target); } fctl.update(); if (bctl._userMoved) bctl.update();
		fr.render(fs, fcam); br.render(bs, bcam);
	}
	requestAnimationFrame(loop);
	(function pollLoad() { var t = $('connectomeSubtitle').textContent; if (t) $('loadText').textContent = t; var m = /(\d+)%/.exec(t); if (m) $('loadBar').style.width = m[1] + '%'; if (!S.ready) setTimeout(pollLoad, 200); })();

	/* ---------------- fast-forward training (S): untrained brain -> trained, as batches of worker ticks ---------------- */
	var FT = { on: false, trial: 0, trials: 150, pace: 150,  /* ms between trials, so learning is watchable (~45 s total) */ PRE: 8, CUE: 20, GAP: 8, DA_TICK: 12, eta: 0.003, joIdx: null, cueSpikes: 0 };
	function groupIndices(name) { var n2i = BRAIN.workerGroupNameToId, garr = BRAIN.workerGroupIdArr, g = n2i[name], out = []; for (var i = 0; i < garr.length; i++) if (garr[i] === g) out.push(i); return Uint32Array.from(out); }
	function ftCue(on) {
		if (on) BRAIN.worker.postMessage({ type: 'setStimulusState', indices: FT.joIdx, intensities: new Float32Array(FT.joIdx.length).fill(STIM * 3) });
		else BRAIN.worker.postMessage({ type: 'setStimulusState', indices: null, intensities: null });
	}
	function ftPostTrial() {   // one whole trial as an ordered batch; the worker answers each 'step' when done
		var w = BRAIN.worker;
		w.postMessage({ type: 'reset' });
		w.postMessage({ type: 'step', n: FT.PRE, tag: 'pre' });
		ftCue(true);
		w.postMessage({ type: 'step', n: FT.DA_TICK, tag: 'cue1' });
		reward(1);                                                          // paired reward at the same cue tick as the offline trainer
		w.postMessage({ type: 'step', n: FT.CUE - FT.DA_TICK, tag: 'cue2' });
		ftCue(false);
		w.postMessage({ type: 'step', n: FT.GAP, tag: 'gap' });
	}
	function fastTrainStart() {
		if (!BRAIN.worker || !S.plasticOn || FT.on) return;
		audio(); S.active = false; S.performing = false; setCue(false);
		setTrained(false);                                  // back to the original FlyWire weights; plastic set and dopamine rule stay armed
		BRAIN.worker.postMessage({ type: 'plasticity', postGroups: [], daGroups: [], edges: trained.edges || undefined, eta: FT.eta });
		FT.joIdx = FT.joIdx || groupIndices('MECH_JO');
		FT.on = true; FT.trial = 0; FT.cueSpikes = 0; S.attempt = 0; S.skill = 0; S.evokedEMA = 0; S.evoked = 0; S.celebrated = false;
		BRAIN.stimulate.manual = true;
		BRAIN.worker.postMessage({ type: 'fast', on: true, groups: Object.keys(motorGid || {}).map(Number) });
		$('btnTrain').classList.add('on');
		BRAIN.onStepped = ftStepped;
		ftPostTrial();
	}
	function fastTrainStop(finished) {
		if (!FT.on) return; FT.on = false; BRAIN.onStepped = null;
		ftCue(false); BRAIN.worker.postMessage({ type: 'fast', on: false }); BRAIN.worker.postMessage({ type: 'reset' });
		BRAIN.stimulate.manual = false; $('btnTrain').classList.remove('on'); $('cueState').textContent = '';
		trained.on = finished; var pl = BRAIN.plasticity;
		$('brainState').textContent = finished ? 'trained live · ' + (pl ? pl.edges.toLocaleString() : '') + ' synapses · Δw +' + (pl && pl.sumW0 ? (pl.sumDelta / pl.sumW0 * 100).toFixed(0) : '0') + '%' : 'untrained · training stopped';
		$('brainState').style.color = finished ? '#e0b04a' : '';
		if (finished) setTimeout(function () { if (!FT.on && !S.performing) perform(); }, 2500);   // straight into the shofar service with the brain just trained
	}
	function ftStepped(d) {
		if (!FT.on) return;
		var sum = 0; for (var i = 0; i < d.motor.length; i++) sum += d.motor[i];
		if (d.tag === 'pre') { FT.cueSpikes = 0; $('cueState').textContent = '♪ sound cue → ear neurons'; }
		else if (d.tag === 'cue1' || d.tag === 'cue2') FT.cueSpikes += sum;
		else if (d.tag === 'gap') {
			$('cueState').textContent = '';
			FT.trial++; S.attempt = FT.trial;
			S.evoked = FT.cueSpikes / FT.CUE; S.evokedEMA = S.evokedEMA ? S.evokedEMA * 0.7 + S.evoked * 0.3 : S.evoked; S.lastEffort = S.evoked;
			S.skill = skillOf(S.evokedEMA);
			var q = skillOf(S.evoked);
			if (FT.trial % 10 === 1 || FT.trial === FT.trials) playBlast(q < 0.3 ? 'sputter' : stageFor(S.skill), Math.min(1, 0.12 + q * 0.95));
			if (FT.trial >= FT.trials) { fastTrainStop(true); return; }
			setTimeout(function () { if (FT.on) ftPostTrial(); }, FT.pace);
		}
	}

	/* ---------------- controls ---------------- */
	function toggle() { if (FT.on) fastTrainStop(false); else fastTrainStart(); }
	function perform() { if (FT.on) fastTrainStop(false); audio(); if (BRAIN.worker) BRAIN.worker.postMessage({ type: 'reset' }); S.performing = true; S.perfIdx = 0; S.active = true; S.phase = 'idle'; S.phaseT = 0; S.celebrated = true; $('btnTrain').classList.remove('on'); }
	function reset() { S.skill = 0; S.evokedEMA = 0; S.evoked = 0; S.attempt = 0; S.celebrated = false; S.lastEffort = 0; S.performing = false; S.active = false; setCue(false); $('btnTrain').classList.remove('on'); $('blast').classList.remove('on'); }
	$('btnTrain').onclick = toggle; $('btnPerform').onclick = perform; $('btnReset').onclick = reset;
	$('btnBlow').onclick = function () { audio(); playBlast(stageFor(S.skill), Math.max(0.2, S.skill)); };
	document.addEventListener('keydown', function (e) { var k = e.key.toLowerCase(); if (k === 's') toggle(); else if (k === 'p' || k === 'b') perform(); else if (k === 'r') reset(); else if (k === 't') { setTrained(!trained.on); } else if (k === 'h') { document.querySelectorAll('.hud,#controls,.wm').forEach(function (el) { el.style.visibility = el.style.visibility === 'hidden' ? '' : 'hidden'; }); } });
	window.Shofar = { state: S, toggle: toggle, perform: perform, reset: reset, toggleTrained: function () { setTrained(!trained.on); } };
})();
