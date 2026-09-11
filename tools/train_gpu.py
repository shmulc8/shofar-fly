"""GPU (Apple MPS) port of the FlyBrain LIF simulator for offline conditioning.

Same connectome, same LIF rules as js/sim-worker.js (leak 0.95, threshold 1.0, refractory 3 ticks,
weights normalized by max then clamp(w*gain, ±cap)), same three-factor dopamine-gated plasticity.
One simplification: no neuropil gating (every neuron is integrated every tick; the worker only
skips idle groups to save CPU).

Trial = PRE ticks silence -> CUE ticks MECH_JO stimulated -> GAP ticks. Evoked = motor(cue) - motor(pre).
Arms: reward (contingent dopamine), noreward, yoked (random dopamine), static (no plasticity).
Exports stage/trained_weights.bin (edge indices in the worker's sorted CSR order) + stage/trained_meta.json.

usage: python tools/train_gpu.py --trials 300 --arms reward,noreward,yoked,static
"""
import argparse, gzip, json, os, struct, sys, time
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
P = argparse.ArgumentParser()
P.add_argument('--trials', type=int, default=300)
P.add_argument('--arms', default='reward,noreward,yoked,static')
P.add_argument('--gain', type=float, default=100.0)
P.add_argument('--cap', type=float, default=0.6)
P.add_argument('--eta', type=float, default=0.003)
P.add_argument('--da-floor', type=float, default=25.0, help='dopamine spikes/tick below this count as background, not reward')
P.add_argument('--da-scale', type=float, default=100.0)
P.add_argument('--da0', type=float, default=0.04, help='dopamine baseline: below it co-active synapses depress (LTD), above it potentiate (LTP)')
P.add_argument('--min-gain', type=float, default=0.25)
P.add_argument('--adapt-inc', type=float, default=0.0, help='spike-frequency adaptation: threshold increment per spike (0 = off)')
P.add_argument('--adapt-decay', type=float, default=0.9)
P.add_argument('--no-tonic', action='store_true', help='no background drive into CX groups')
P.add_argument('--reset-between', action='store_true', help='hard-reset V/refractory/adaptation/traces between trials (weights persist)')
P.add_argument('--baseline', type=int, default=20, help='CS-only trials before training, plasticity off')
P.add_argument('--test', type=int, default=20, help='CS-only trials after training, plasticity off (the measurement)')
P.add_argument('--da-tick', type=int, default=12, help='cue tick at which the dopamine pulse is delivered (pair arm)')
P.add_argument('--da-amp', type=float, default=2.7, help='dopamine one-shot amplitude (single stim call)')
P.add_argument('--plastic-post', default='SEZ_FEED,MN_PROBOSCIS,MN_HEAD,MB_MBON_APP')
P.add_argument('--cue-path', action='store_true', help='restrict plastic presynaptic neurons to those firing on >= cue-path-min of 10 CS-only probe trials')
P.add_argument('--cue-path-min', type=float, default=0.3)
P.add_argument('--da-mode', default='signal', choices=['signal', 'spikes'], help="signal: reward raises the dopamine trace directly (modulatory third factor); spikes: stimulate MB_DAN_REW neurons")
P.add_argument('--da-signal', type=float, default=1.0, help='dopamine trace increment per reward in signal mode')
P.add_argument('--parallel', action='store_true', help='run each arm in its own process concurrently (one GPU queue per arm)')
P.add_argument('--gap', type=int, default=40, help='silent ticks after the cue')
P.add_argument('--backward-tick', type=int, default=30, help='gap tick at which the backward arm gets its reward')
P.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
A = P.parse_args()
dev = torch.device(A.device)

LEAK, THR, REFR, WEIGHT_SCALE = 0.95, 1.0, 3, 0.15
PRE, CUE, GAP = 40, 20, A.gap
STIM = 0.15; JO_INT = STIM * 3; TONIC = 0.08; DA_INT = STIM * 6
MOTOR = ['MN_PROBOSCIS', 'MN_HEAD', 'SEZ_FEED']
READOUTS = {'mbon': ['MB_MBON_APP'], 'feed': ['SEZ_FEED'], 'mouth': ['MN_PROBOSCIS', 'MN_HEAD'], 'kc': ['MB_KC']}
PLASTIC_POST = A.plastic_post.split(',')
PRE_DECAY, DA_DECAY, MAX_GAIN = 0.92, 0.85, 3.0

# ---------------- load connectome exactly like the worker ----------------
meta = json.load(open(os.path.join(ROOT, 'data/neuron_meta.json')))
gname = [g['name'] for g in meta['groups']]; gid_of = {g['name']: g['id'] for g in meta['groups']}
raw = gzip.open(os.path.join(ROOT, 'data/connectome.bin.gz')).read()
N, E = struct.unpack_from('<II', raw, 0)
edges = np.frombuffer(raw, dtype=np.dtype([('pre', '<u4'), ('post', '<u4'), ('w', '<f4')]), count=E, offset=8)
meta_off = 8 + E * 12
per = np.frombuffer(raw, dtype=np.dtype([('region', 'u1'), ('group', '<u2')]), count=N, offset=meta_off)
group_orig = per['group'].astype(np.int64)
w_norm = edges['w'].astype(np.float32) / np.abs(edges['w']).max() * WEIGHT_SCALE

# worker order: neurons stable-sorted by group; edges: CSR rows in sorted order, file order within a row
sorted_by_group = np.argsort(group_orig, kind='stable')          # sorted_pos -> original
orig2sorted = np.empty(N, np.int64); orig2sorted[sorted_by_group] = np.arange(N)
pre_s = orig2sorted[edges['pre'].astype(np.int64)]; post_s = orig2sorted[edges['post'].astype(np.int64)]
order = np.argsort(pre_s, kind='stable')                           # worker edge index -> file edge index
pre_s, post_s, w_norm = pre_s[order], post_s[order], w_norm[order]
group_s = group_orig[sorted_by_group]
print(f'{N:,} neurons, {E:,} edges, device={dev}', flush=True)

def idx_of(names):
    m = np.zeros(N, bool)
    for n in names: m |= (group_s == gid_of[n])
    return torch.from_numpy(np.nonzero(m)[0]).to(dev)

G = {n: idx_of([n]) for n in ['MECH_JO', 'CX_FC', 'CX_EPG', 'CX_PFN', 'MB_DAN_REW'] + MOTOR}
motor_idx = idx_of(MOTOR); da_idx = G['MB_DAN_REW']
RO = {k: idx_of(v) for k, v in READOUTS.items()}
RO_KEYS = ['motor'] + list(READOUTS)
_ro_gid = np.full(N, len(RO_KEYS), np.int64)
for _k, _names in list(READOUTS.items()):
    for _n in _names: _ro_gid[group_s == gid_of[_n]] = RO_KEYS.index(_k)
ro_gid = torch.from_numpy(_ro_gid).to(dev)
motor_mask = torch.zeros(N, dtype=torch.bool, device=dev); motor_mask[motor_idx] = True
def counts(fired):  # per-tick readout counts as a GPU tensor [len(RO_KEYS)]
    c = torch.bincount(ro_gid[fired], minlength=len(RO_KEYS) + 1)[:len(RO_KEYS)]
    c[0] = (fired & motor_mask).sum(); return c
pre_t = torch.from_numpy(pre_s).to(dev); post_t = torch.from_numpy(post_s).to(dev)
w_base = torch.clamp(torch.from_numpy(w_norm).to(dev) * A.gain, -A.cap, A.cap)
def build_plastic(pre_mask=None):
    m = np.isin(group_s[post_s], [gid_of[n] for n in PLASTIC_POST]) & (group_s[pre_s] != gid_of['MB_DAN_REW'])
    if pre_mask is not None: m &= pre_mask[pre_s]
    idx = torch.nonzero(torch.from_numpy(m).to(dev) & (w_base > 0)).squeeze(1)
    print(f'plastic edges: {idx.numel():,}' + ('' if pre_mask is None else f' (pre restricted to {int(pre_mask.sum()):,} cue-path neurons)'), flush=True)
    return idx
plastic_idx = build_plastic()

class Sim:
    def __init__(self, plastic):
        self.w = w_base.clone(); self.w0 = self.w[plastic_idx].clone()
        self.V = torch.zeros(N, device=dev); self.fired = torch.zeros(N, dtype=torch.bool, device=dev)
        self.refr = torch.zeros(N, dtype=torch.int32, device=dev)
        self.pre_tr = torch.zeros(N, device=dev); self.post_tr = torch.zeros(N, device=dev)
        self.da = 0.0; self.da_pulse = 0.0; self.plastic = plastic; self.sustained = None; self.oneshot = []
        self.dsum = torch.zeros((), device=dev)
        self.adapt = torch.zeros(N, device=dev)

    def set_sustained(self, segs):
        if not segs: self.sustained = None; return
        idx = torch.cat([G[n] for n, _ in segs]); val = torch.cat([torch.full((G[n].numel(),), v, device=dev) for n, v in segs])
        self.sustained = (idx, val)

    def stim(self, name, v): self.oneshot.append((G[name], v))
    def reward(self, scale=1.0):
        if A.da_mode == 'spikes': self.stim('MB_DAN_REW', A.da_amp * scale)
        else: self.da_pulse += A.da_signal * scale
    def reset_state(self):
        self.V.zero_(); self.refr.zero_(); self.adapt.zero_(); self.fired.zero_(); self.pre_tr.zero_(); self.post_tr.zero_(); self.da = 0.0; self.da_pulse = 0.0; self.oneshot = []

    @torch.no_grad()
    def tick(self):
        active = self.refr == 0
        self.V = torch.where(active, self.V * LEAK, torch.zeros_like(self.V))
        self.refr = torch.clamp(self.refr - 1, min=0)
        if self.sustained is not None:
            idx, val = self.sustained
            self.V.index_add_(0, idx, val * active[idx].float())
        for idx, v in self.oneshot: self.V.index_add_(0, idx, torch.full((idx.numel(),), v, device=dev))
        self.oneshot = []
        contrib = self.w * self.fired[pre_t].float()
        self.V.index_add_(0, post_t, contrib)
        self.adapt = self.adapt * A.adapt_decay
        new_fired = (self.refr == 0) & (self.V >= THR + self.adapt)
        if A.adapt_inc > 0: self.adapt = torch.where(new_fired, self.adapt + A.adapt_inc, self.adapt)
        self.V = torch.where(new_fired, torch.zeros_like(self.V), self.V)
        self.refr = torch.where(new_fired, torch.full_like(self.refr, REFR), self.refr)
        self.fired = new_fired
        if self.plastic:
            f = new_fired.float()
            self.pre_tr = torch.where(new_fired, torch.ones_like(self.pre_tr), self.pre_tr * PRE_DECAY)
            self.post_tr = torch.where(new_fired, torch.ones_like(self.post_tr), self.post_tr * PRE_DECAY)
            spike_da = min(1.0, max(0.0, f[da_idx].sum().item() - A.da_floor) / A.da_scale) if A.da_mode == 'spikes' else 0.0
            self.da = self.da * DA_DECAY + spike_da + self.da_pulse; self.da_pulse = 0.0
            # three-factor rule, dopamine relative to baseline: LTP when DA > da0, LTD when DA < da0
            tr = self.pre_tr[pre_t[plastic_idx]] * self.post_tr[post_t[plastic_idx]]
            cur = self.w[plastic_idx]
            nw = torch.clamp(cur + A.eta * (self.da - A.da0) * tr * self.w0, self.w0 * A.min_gain, self.w0 * MAX_GAIN)
            self.dsum += (nw - cur).sum()
            self.w[plastic_idx] = nw
        return new_fired

    def motor(self, fired): return fired[motor_idx].sum().item()
    def readout(self, fired): return {k: fired[v].sum().item() for k, v in RO.items()}

TONIC_SEGS = [] if A.no_tonic else [('CX_FC', TONIC), ('CX_EPG', TONIC), ('CX_PFN', TONIC)]

def cue_path_mask(n_probe=10):
    sim = Sim(plastic=False); cnt = torch.zeros(N, device=dev)
    for _ in range(n_probe):
        sim.reset_state(); sim.set_sustained(TONIC_SEGS)
        for _ in range(PRE): sim.tick()
        sim.set_sustained(TONIC_SEGS + [('MECH_JO', JO_INT)]); anyf = torch.zeros(N, dtype=torch.bool, device=dev)
        for _ in range(CUE): anyf |= sim.tick()
        cnt += anyf.float()
    m = (cnt >= A.cue_path_min * n_probe).cpu().numpy()
    top = {}
    for g in np.unique(group_s[m]): top[gname[g]] = int((group_s[m] == g).sum())
    print('cue-path neurons by group: ' + ' '.join(f'{k}={v}' for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:12]), flush=True)
    return m

def trial(sim, arm, phase):
    """One trial. phase: 'base' | 'train' | 'test'. Dopamine is delivered only during 'train'. All readouts stay on the GPU until the end."""
    if A.reset_between: sim.reset_state()
    sim.plastic = sim.has_plastic and phase == 'train'
    sim.set_sustained(TONIC_SEGS)
    nk = len(RO_KEYS); pre_c = torch.zeros(nk, device=dev); cue_c = torch.zeros(nk, device=dev)
    for i in range(PRE):
        if phase == 'train' and arm == 'unpaired' and i == 2: sim.reward()   # same dose, 38 ticks before the cue
        pre_c += counts(sim.tick())
    sim.set_sustained(TONIC_SEGS + [('MECH_JO', JO_INT)])
    anyf = torch.zeros(N, dtype=torch.bool, device=dev); lat = torch.full((), -1.0, device=dev); half = None
    for i in range(CUE):
        f = sim.tick(); c = counts(f); cue_c += c; anyf |= f
        lat = torch.where((lat < 0) & (c[0] > 0), torch.full((), float(i), device=dev), lat)
        if phase == 'train' and arm == 'pair' and i == A.da_tick: sim.reward()
        if i == CUE // 2 - 1: half = cue_c.clone()
        if phase == 'train' and arm == 'reward' and i == CUE // 2:
            ev = (half[0] / (CUE / 2) - pre_c[0] / PRE).item()
            if ev > 0: sim.reward(min(1.5, 0.3 + ev / 4))
    sim.set_sustained(TONIC_SEGS)
    for i in range(GAP):
        sim.tick()
        if phase == 'train' and arm == 'backward' and i == A.backward_tick: sim.reward()  # reward well after the cue ended
    nrec = (anyf & motor_mask).sum()
    vals = torch.cat([pre_c / PRE, cue_c / CUE, lat.reshape(1), nrec.reshape(1).float()]).cpu().numpy()  # the one sync per trial
    pre_v, cue_v = vals[:nk], vals[nk:2 * nk]
    row = {'phase': phase, 'pre': float(pre_v[0]), 'cue': float(cue_v[0]), 'evoked': float(cue_v[0] - pre_v[0]), 'nrec': int(vals[-1]), 'lat': float(vals[-2])}
    for j, k in enumerate(RO_KEYS[1:], 1): row['ev_' + k] = float(cue_v[j] - pre_v[j]); row['pre_' + k] = float(pre_v[j])
    return row

def run_arm(arm):
    sim = Sim(plastic=(arm != 'static')); sim.has_plastic = sim.plastic
    sim.set_sustained(TONIC_SEGS)
    for _ in range(40): sim.tick()
    curve = []; t0 = time.time()
    for _ in range(A.baseline): curve.append(trial(sim, arm, 'base'))
    for t in range(A.trials):
        curve.append(trial(sim, arm, 'train'))
        if (t + 1) % 25 == 0:
            print(f'  {arm} trial {t+1}: evoked(last25)={np.mean([c["evoked"] for c in curve[-25:]]):.2f} dW={sim.dsum.item() / sim.w0.sum().item() * 100:.2f}%  {(time.time()-t0)/(t+1+A.baseline)*1000:.0f} ms/trial', flush=True)
    for _ in range(A.test): curve.append(trial(sim, arm, 'test'))
    base = [c for c in curve if c['phase'] == 'base'] or curve[:10]; test = [c for c in curve if c['phase'] == 'test'] or curve[-10:]
    mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
    dw = sim.dsum.item() / sim.w0.sum().item() * 100 if sim.has_plastic else 0.0
    b, e = mean(base, 'evoked'), mean(test, 'evoked')
    print(f'{arm:9s} evoked base={b:.2f} test={e:.2f} ({(e/b-1)*100 if b else 0:+.0f}%)  recruited {mean(base,"nrec"):.1f}->{mean(test,"nrec"):.1f} of {motor_idx.numel()} motor neurons  latency {mean(base,"lat"):.1f}->{mean(test,"lat"):.1f}  rest(test)={mean(test,"pre"):.2f}  dW={dw:.2f}%', flush=True)
    for r in RO: print(f'           {r:6s} evoked base={mean(base,"ev_"+r):7.2f} test={mean(test,"ev_"+r):7.2f}   rest base={mean(base,"pre_"+r):6.2f} test={mean(test,"pre_"+r):6.2f}', flush=True)
    json.dump(curve, open(os.path.join(ROOT, f'tools/out_gpu_{arm}.json'), 'w'))
    if arm in ('reward', 'pair'):
        idx = plastic_idx.cpu().numpy().astype('<u4'); w = sim.w[plastic_idx].cpu().numpy().astype('<f4')
        os.makedirs(os.path.join(ROOT, 'stage'), exist_ok=True)
        with open(os.path.join(ROOT, 'stage/trained_weights.bin'), 'wb') as f:
            f.write(struct.pack('<I', len(idx))); f.write(idx.tobytes()); f.write(w.tobytes())
        json.dump({'gain': A.gain, 'cap': A.cap, 'eta': A.eta, 'trials': A.trials, 'baseline': A.baseline, 'test': A.test, 'edges': int(len(idx)), 'device': str(dev),
                   'evoked_base': b, 'evoked_test': e, 'recruited_base': mean(base, 'nrec'), 'recruited_test': mean(test, 'nrec'), 'dW_percent': dw,
                   'da0': A.da0, 'da_mode': A.da_mode, 'da_signal': A.da_signal, 'da_amp': A.da_amp, 'da_tick': A.da_tick, 'min_gain': A.min_gain, 'adapt_inc': A.adapt_inc, 'adapt_decay': A.adapt_decay, 'no_tonic': A.no_tonic,
                   'jo_int': JO_INT, 'reset_between': A.reset_between, 'plastic_post': PLASTIC_POST, 'cue_path': A.cue_path},
                  open(os.path.join(ROOT, 'stage/trained_meta.json'), 'w'), indent=1)
        print(f'saved stage/trained_weights.bin ({len(idx):,} edges)', flush=True)

arms = A.arms.split(',')
if A.parallel and len(arms) > 1:
    import subprocess
    base = [a for a in sys.argv[1:] if a != '--parallel']
    if '--arms' in base: k = base.index('--arms'); del base[k:k + 2]
    procs = [subprocess.Popen([sys.executable, __file__] + base + ['--arms', a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for a in arms]
    for a, pr in zip(arms, procs):
        out = pr.communicate()[0]
        print('\n'.join(l for l in out.splitlines() if 'Warning' not in l and not l.startswith('plastic') and not l.startswith('cue-path') and 'neurons,' not in l), flush=True)
    sys.exit(max(pr.returncode for pr in procs))
if A.cue_path: plastic_idx = build_plastic(cue_path_mask())
for arm in arms:
    run_arm(arm)
