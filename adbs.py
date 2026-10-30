"""Closed-loop adaptive deep-brain-stimulation (aDBS) in a spiking E/I network.

A delayed excitatory-inhibitory LIF network (STN/GPe-like loop) is driven into a
pathological beta-band (13-30 Hz) synchronous regime. Three stimulation policies
are compared: none, open-loop continuous high-frequency DBS, and a closed-loop
controller that estimates beta power online from a LFP proxy and gates/scales
the pulse train. Pure NumPy/SciPy, deterministic given a seed.
"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, welch

DT = 0.1            # ms
T_MS = 10000.0       # simulated time
NE, NI = 400, 100
TAU_M, V_TH, V_RESET, V_REST = 20.0, 1.0, 0.0, 0.0
REF = 2.0           # ms
TAU_SE, TAU_SI = 3.0, 6.0
DELAY = 3.0         # ms, loop delay -> sets oscillation (~15 Hz)
J_EE, J_EI, J_IE, J_II = 0.010, 0.030, -0.060, -0.010
BETA = (13.0, 30.0)
MU_E = 1.15
LOW_DRIVE = 0.95   # weak-beta state

def drive_schedule(t_ms):
    """Pathological state fluctuates (e.g. medication cycles): strong-beta
    episodes (MU high) alternate with weak-beta episodes (MU low)."""
    return 1.15 if (t_ms // 2500) % 2 == 1 else LOW_DRIVE

def simulate(policy="none", seed=1, dbs_amp=4.0, dbs_freq=130.0,
             setpoint=0.72, ki=4e-4, amax=8.0, thr_on=0.75, thr_off=0.68):
    rng = np.random.default_rng(seed)
    n = NE + NI
    steps = int(T_MS / DT)
    dly = int(DELAY / DT)
    v = rng.uniform(0, 1, n)
    ref = np.zeros(n)
    sE = np.zeros(n); sI = np.zeros(n)
    spk_hist = np.zeros((dly, n), dtype=bool)
    mu = np.r_[np.full(NE, MU_E), np.full(NI, 0.9)]        # tonic drive
    mu_E_idx = slice(0, NE)
    noise = 0.9
    pop_rate = np.zeros(steps); lfp = np.zeros(steps)
    stim_on = np.zeros(steps); stim_amp_t = np.zeros(steps); env_t = np.zeros(steps)
    spikes_E = []
    # online beta estimator: band-pass -> rectified envelope (leaky average)
    fs = 1000.0 / DT
    sos = butter(2, BETA, btype="band", fs=fs / 50, output="sos")  # on 50-step bins
    zi = sosfilt_zi(sos)[:, :] * 0
    env = 0.0; env_alpha = 0.02
    bin_acc = 0.0; bin_n = 0
    pulse_period = int(1000.0 / dbs_freq / DT)
    pulse_w = int(0.3 / DT) * 2
    gate = policy == "open"; amp = dbs_amp if gate else 0.0
    ctrl_every = 50
    for t in range(steps):
        prev = spk_hist[t % dly]
        eS = prev[:NE].sum(); iS = prev[NE:].sum()
        sE += -sE * DT / TAU_SE + J_EE * eS * np.r_[np.ones(NE), np.zeros(NI)] \
              + J_EI * eS * np.r_[np.zeros(NE), np.ones(NI)]
        sI += -sI * DT / TAU_SI + J_IE * iS * np.r_[np.ones(NE), np.zeros(NI)] \
              + J_II * iS * np.r_[np.zeros(NE), np.ones(NI)]
        mu[mu_E_idx] = drive_schedule(t * DT)
        I = mu + sE + sI + noise * rng.standard_normal(n) / np.sqrt(TAU_M / DT) * 3
        # stimulation: biphasic-like pulse train onto excitatory pop (clipped, shunting-free)
        pulse = (t % pulse_period) < pulse_w
        if pulse and amp > 0:
            I[:NE] -= amp * 0.5          # net effect of HF DBS: desynchronising
            I[:NE] += amp * rng.standard_normal(NE) * 1.5
        stim_on[t] = pulse and amp > 0; stim_amp_t[t] = amp; env_t[t] = env
        v += DT / TAU_M * (V_REST - v + I * TAU_M * 0.05 * 20 / 20)
        v = np.where(ref > 0, V_RESET, v)
        ref = np.maximum(ref - DT, 0)
        s = v >= V_TH
        v[s] = V_RESET; ref[s] = REF
        spk_hist[t % dly] = s
        pop_rate[t] = s[:NE].sum()
        lfp[t] = sE[:NE].mean() - sI[:NE].mean()
        if s[:NE].any():
            spikes_E.append((t * DT, np.flatnonzero(s[:NE])))
        # ---- controller ----
        bin_acc += lfp[t]; bin_n += 1
        if bin_n == 50:
            x = bin_acc / 50; bin_acc = 0.0; bin_n = 0
            y, zi = sosfilt(sos, [x], zi=zi)
            env = (1 - env_alpha) * env + env_alpha * abs(y[0])
            if policy == "gated":
                if env > thr_on: amp = dbs_amp
                elif env < thr_off: amp = 0.0
            if policy == "closed":
                # integral controller: drive beta envelope to the setpoint
                amp = float(np.clip(amp + ki * (env - setpoint) * 50 * DT * 10, 0.0, amax))
    return dict(rate=pop_rate, lfp=lfp, stim=stim_on, amp=stim_amp_t, env=env_t, spikes=spikes_E)

def beta_power(lfp, t0_ms=1000.0):
    x = lfp[int(t0_ms / DT):]
    x = x.reshape(-1, 10).mean(1)           # 1 ms bins -> 1 kHz
    f, p = welch(x - x.mean(), fs=1000.0, nperseg=2048)
    m = (f >= BETA[0]) & (f <= BETA[1])
    return float(np.trapezoid(p[m], f[m])), f, p

def beta_burst_fraction(lfp, t0_ms=1000.0):
    x = lfp[int(t0_ms / DT):].reshape(-1, 10).mean(1)
    sos = butter(2, BETA, btype="band", fs=1000.0, output="sos")
    from scipy.signal import hilbert
    e = np.abs(hilbert(sosfilt(sos, x - x.mean())))
    return float((e > np.percentile(e, 75)).mean())
