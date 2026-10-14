import time, math
import numpy as np, scipy.io as sio, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t0 = time.time()
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
torch.manual_seed(0); np.random.seed(0)

m = sio.loadmat('SNLS80mV.mat')
print([k for k in m.keys() if not k.startswith('__')])
if 'V1' not in m or 'V2' not in m:
    raise SystemExit("V1 or V2 missing")
u_all = np.asarray(m['V1']).ravel().astype(np.float64)
y_all = np.asarray(m['V2']).ravel().astype(np.float64)
assert len(u_all) == 131072 and len(y_all) == 131072
u = u_all[42650:46746]; y = y_all[42650:46746]
assert len(u) == 4096
u = u - u.mean(); y = y - y.mean()
dt = 1.0 / 610.35
NTR = 3072
print("max|y| train", np.abs(y[:NTR]).max(), "test", np.abs(y[NTR:]).max(), "std y", y.std(), "std u", u.std())

ytr, utr = y[:NTR], u[:NTR]
Sy = ytr.std(); w0 = 2 * math.pi * 70; Sv = w0 * Sy; Su = utr.std(); Sa = w0 ** 2 * Sy

# LS init
acc = (ytr[2:] - 2 * ytr[1:-1] + ytr[:-2]) / dt ** 2
vc = (ytr[2:] - ytr[:-2]) / (2 * dt)
yc = ytr[1:-1]; uc = utr[1:-1]
A = np.stack([-vc, -yc, -yc ** 3, uc], 1)
coef = np.linalg.lstsq(A, acc, rcond=None)[0]
a0, b0, c0, g0 = coef
print("LS fit", coef)
if a0 <= 0: a0 = 30.0
if b0 <= 0 or b0 > 1e7 or b0 < 1e4 or c0 <= 0 or c0 > 1e8:
    b0 = w0 ** 2; c0 = 5e5
print("init a,b,c,g", a0, b0, c0, g0)

Y = torch.tensor(y); U = torch.tensor(u)
Vfull = torch.zeros(len(y))
Vfull[1:-1] = (Y[2:] - Y[:-2]) / (2 * dt)

class Acc(nn.Module):
    def __init__(self):
        super().__init__()
        self.pa = nn.Parameter(torch.tensor(1.0)); self.pb = nn.Parameter(torch.tensor(1.0))
        self.pc = nn.Parameter(torch.tensor(1.0)); self.pg = nn.Parameter(torch.tensor(1.0))
        self.l1 = nn.Linear(3, 32); self.l2 = nn.Linear(32, 32); self.l3 = nn.Linear(32, 1)
        for l in (self.l1, self.l2): nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
        nn.init.zeros_(self.l3.weight); nn.init.zeros_(self.l3.bias)
    def phys(self):
        return self.pa * a0, self.pb * b0, self.pc * c0, self.pg * g0
    def forward(self, yy, vv, uu):
        a, b, c, g = self.phys()
        x = torch.stack([yy / Sy, vv / Sv, uu / Su], -1)
        n = self.l3(torch.tanh(self.l2(torch.tanh(self.l1(x))))).squeeze(-1)
        return -a * vv - b * yy - c * yy ** 3 + g * uu + Sa * n, n

def rk4_step(model, z, u0, u1):
    um = 0.5 * (u0 + u1)
    y_, v_ = z[:, 0], z[:, 1]
    a1, n1 = model(y_, v_, u0)
    k1 = (v_, a1)
    a2, _ = model(y_ + 0.5 * dt * k1[0], v_ + 0.5 * dt * k1[1], um)
    k2 = (v_ + 0.5 * dt * k1[1], a2)
    a3, _ = model(y_ + 0.5 * dt * k2[0], v_ + 0.5 * dt * k2[1], um)
    k3 = (v_ + 0.5 * dt * k2[1], a3)
    a4, _ = model(y_ + dt * k3[0], v_ + dt * k3[1], u1)
    k4 = (v_ + dt * k3[1], a4)
    yn = y_ + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    vn = v_ + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return torch.stack([yn, vn], 1), (n1 ** 2).mean()

def rollout(model, z0, useq):
    L = useq.shape[1] - 1
    z = z0; ys = [z[:, 0]]; reg = 0.0
    for k in range(L):
        z, r = rk4_step(model, z, useq[:, k], useq[:, k + 1])
        ys.append(z[:, 0]); reg = reg + r
    return torch.stack(ys, 1), reg / L

def window_batch(L, B):
    s = torch.randint(1, NTR - L, (B,))
    idx = s[:, None] + torch.arange(L + 1)[None, :]
    z0 = torch.stack([Y[s], Vfull[s]], 1)
    return z0, U[idx], Y[idx]

LAM = 1e-3; W = 16; GCAP = 230.0
model = Acc()
def pstr():
    a, b, c, g = [float(p) for p in model.phys()]
    return f"a={a:.4g} b={b:.5g} c={c:.5g} g={g:.5g}"

def train_stage(name, L, B, iters, lr_net, lr_phys, budget, freeze_net):
    phys = [model.pa, model.pb, model.pc, model.pg]
    net = [p for n_, p in model.named_parameters() if n_.startswith('l')]
    groups = [{'params': phys, 'lr': lr_phys}]
    if not freeze_net: groups.append({'params': net, 'lr': lr_net})
    opt = torch.optim.Adam(groups)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    ts = time.time()
    for it in range(iters):
        if time.time() - ts > budget or time.time() - t0 > GCAP:
            print(name, "budget stop at it", it); break
        z0, ub, yb = window_batch(L, B)
        yh, reg = rollout(model, z0, ub)
        w = min(W, L - 1)
        loss = (((yh[:, w:] - yb[:, w:]) / Sy) ** 2).mean() + LAM * reg
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if it % 50 == 0:
            print(f"{name} it{it} loss={loss.item():.5g} t={time.time()-t0:.1f}s {pstr()}", flush=True)
    print(name, "done", f"t={time.time()-t0:.1f}s", pstr(), flush=True)

train_stage("S0", 64, 64, 200, 0, 1e-2, 20, True)
train_stage("S1", 32, 128, 500, 3e-3, 1e-2, 40, False)
train_stage("S2", 128, 64, 400, 3e-3, 1e-2, 70, False)
train_stage("S3", 512, 16, 150, 3e-3, 1e-2, 70, False)

def full_rollout(n):
    z0 = torch.tensor([[y[0], (y[1] - y[0]) / dt]])
    return rollout(model, z0, U[None, :n])[0][0]

if time.time() - t0 < 180:
    print("Stage 4 full train rollout")
    ts = time.time()
    opt = torch.optim.Adam([p for p in model.parameters()], lr=1e-3)
    for it in range(20):
        if time.time() - ts > 30 or time.time() - t0 > GCAP: break
        yh = full_rollout(NTR)
        loss = (((yh[W:] - Y[:NTR][W:]) / Sy) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(f"S4 it{it} loss={loss.item():.5g} t={time.time()-t0:.1f}s", flush=True)

with torch.no_grad():
    yh = full_rollout(len(y)).numpy()
tr = math.sqrt(np.mean((yh[:NTR] - y[:NTR]) ** 2))
te = math.sqrt(np.mean((yh[NTR:] - y[NTR:]) ** 2))
print("max|yhat|", np.abs(yh).max(), "zero-predictor test RMSE", math.sqrt(np.mean(y[NTR:] ** 2)))
tt = np.arange(len(y) - NTR) * dt
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(tt, y[NTR:], label='measured'); ax[0].plot(tt, yh[NTR:], '--', label='predicted')
ax[0].set_ylabel('y [V]'); ax[0].legend(); ax[0].set_title(f'Free run, test segment, TEST_RMSE={te:.4g} V')
ax[1].plot(tt, yh[NTR:] - y[NTR:]); ax[1].set_ylabel('error [V]'); ax[1].set_xlabel('time [s]')
plt.tight_layout(); plt.savefig('test_segment.png', dpi=120)
print("final", pstr(), f"runtime={time.time()-t0:.1f}s")
print(f"TRAIN_RMSE={tr}")
print(f"TEST_RMSE={te}")
