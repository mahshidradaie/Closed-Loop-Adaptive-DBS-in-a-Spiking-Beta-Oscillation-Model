"""
Closed-loop adaptive deep brain stimulation (aDBS) in a spiking STN-GPe network.

A delayed excitatory-inhibitory loop of leaky integrate-and-fire neurons
(subthalamic nucleus, STN, and external globus pallidus, GPe) produces
pathological beta-band (13-30 Hz) oscillations when the coupling is strong
("parkinsonian" state). Three stimulation policies are compared:

  none        - no stimulation
  continuous  - 130 Hz pulse train applied the whole time (conventional DBS)
  adaptive    - the same pulse train is switched on only while the beta
                envelope of the LFP proxy exceeds a threshold (closed loop)

Outputs: beta power, fraction of time stimulated (a proxy for delivered
energy) and a summary figure.

Run:  python adbs_sim.py
"""
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.25e-3          # s
FS = 1.0 / DT
N_STN, N_GPE = 200, 200


def build_network(rng, p=0.25):
    c_eg = (rng.random((N_GPE, N_STN)) < p).astype(float)   # STN -> GPe
    c_ge = (rng.random((N_STN, N_GPE)) < p).astype(float)   # GPe -> STN
    c_gg = (rng.random((N_GPE, N_GPE)) < p).astype(float)   # GPe -> GPe
    return c_eg / (p * N_STN), c_ge / (p * N_GPE), c_gg / (p * N_GPE)


def simulate(policy, T=6.0, seed=1, w_eg=1.5, w_ge=1.0, w_gg=0.1,
             dbs_amp=6.0, dbs_freq=130.0, thr_on=None, thr_off=None,
             i_stn=1.2, i_gpe=1.2, delay_ms=12.0, min_dwell=0.25, blank=0.2):
    DELAY = int(delay_ms * 1e-3 / DT)  # axonal/synaptic delay (steps)
    rng = np.random.default_rng(seed)
    c_eg, c_ge, c_gg = build_network(rng)
    steps = int(T / DT)
    tau_m, tau_s = 10e-3, 4e-3
    v_stn = rng.random(N_STN) * 0.5
    v_gpe = rng.random(N_GPE) * 0.5
    s_stn = np.zeros(N_STN)      # synaptic drive onto STN
    s_gpe = np.zeros(N_GPE)
    buf_stn = np.zeros((DELAY, N_STN))
    buf_gpe = np.zeros((DELAY, N_GPE))

    i_stn = i_stn + 0.12 * rng.standard_normal(N_STN)   # heterogeneity
    i_gpe = i_gpe + 0.12 * rng.standard_normal(N_GPE)
    lfp = np.zeros(steps)
    stim_on = np.zeros(steps, dtype=bool)
    rate_stn = np.zeros(steps)
    pulse_period = int(round(1.0 / (dbs_freq * DT)))
    pulse_width = 2  # steps
    sos = signal.butter(2, [13, 30], btype="band", fs=FS, output="sos")
    zi = np.zeros((sos.shape[0], 2))
    env = 0.0
    block = int(5e-3 / DT)
    on = (policy == "continuous")
    last_switch = -1e9
    env_rec = np.zeros(steps // block + 1)
    lfp_block = []
    env_alpha = 0.02
    spikes_rec = []

    for k in range(steps):
        pos = k % DELAY
        sp_stn_d = buf_stn[pos]       # spikes emitted DELAY steps ago
        sp_gpe_d = buf_gpe[pos]
        # exponential synapses; a presynaptic spike adds (weight / n_pre_connected)
        s_stn += (-s_stn * DT / tau_s) - w_ge * (c_ge @ sp_gpe_d)
        s_gpe += (-s_gpe * DT / tau_s) + (w_eg * (c_eg @ sp_stn_d)
                                          - w_gg * (c_gg @ sp_gpe_d))

        stim = 0.0
        if on and (k % pulse_period) < pulse_width:
            stim = dbs_amp
        noise_s = 0.6 * rng.standard_normal(N_STN)
        noise_g = 0.6 * rng.standard_normal(N_GPE)
        v_stn += DT / tau_m * (-v_stn + i_stn + s_stn + stim) + np.sqrt(DT / tau_m) * 0.12 * noise_s
        v_gpe += DT / tau_m * (-v_gpe + i_gpe + s_gpe) + np.sqrt(DT / tau_m) * 0.12 * noise_g
        f_s = v_stn >= 1.0
        f_g = v_gpe >= 1.0
        v_stn[f_s] = 0.0
        v_gpe[f_g] = 0.0
        buf_stn[pos] = f_s
        buf_gpe[pos] = f_g
        # LFP proxy: summed synaptic drive onto STN (+ GPe excitation)
        lfp[k] = s_stn.mean() - 0.5 * s_gpe.mean()
        rate_stn[k] = f_s.sum()
        stim_on[k] = on
        lfp_block.append(lfp[k])

        if (k + 1) % block == 0:
            y, zi = signal.sosfilt(sos, np.asarray(lfp_block), zi=zi)
            lfp_block = []
            inst = np.sqrt(np.mean(y ** 2))
            if (k - last_switch) * DT >= blank:   # detector blanking after a switch
                env = (1 - env_alpha * 5) * env + env_alpha * 5 * inst
            env_rec[(k + 1) // block] = env
            if policy == "adaptive" and (k - last_switch) * DT >= min_dwell:
                if (not on) and env > thr_on:
                    on, last_switch = True, k
                elif on and env < thr_off:
                    on, last_switch = False, k
    lp = signal.butter(4, 45, btype="low", fs=FS, output="sos")
    lfp = signal.sosfiltfilt(lp, lfp)
    return dict(lfp=lfp, stim=stim_on, rate=rate_stn, env=env_rec)


def beta_power(x, fs=FS, skip=1.0):
    x = x[int(skip * fs):]
    f, p = signal.welch(x - x.mean(), fs=fs, nperseg=int(2 * fs))
    m = (f >= 13) & (f <= 30)
    return np.trapezoid(p[m], f[m]), f, p


def beta_power_blanked(r, blank=0.1, skip=1.0, edge=0.03):
    """Mean squared 13-30 Hz LFP computed epoch by epoch.

    The recording is cut at every stimulation switch, `blank` seconds after
    each switch are discarded (switch-induced baseline steps are artefacts,
    not rhythm) and every remaining epoch is band-passed separately so that
    the steps cannot leak into the band. Same procedure for all policies."""
    sos = signal.butter(2, [13, 30], btype="band", fs=FS, output="sos")
    x, stim = r["lfp"], r["stim"]
    cuts = [int(skip * FS)] + [k + 1 for k in np.flatnonzero(np.diff(stim.astype(int)) != 0)
                                if k + 1 > int(skip * FS)] + [len(x)]
    tot, n = 0.0, 0
    for a_, b_ in zip(cuts[:-1], cuts[1:]):
        seg = x[a_ + int(blank * FS):b_]
        if len(seg) < int(0.15 * FS):
            continue
        y = signal.sosfiltfilt(sos, seg - seg.mean())
        y = y[int(edge * FS):-int(edge * FS)]
        tot += np.sum(y ** 2)
        n += len(y)
    return tot / n


def run_policies(seed, T=6.0):
    """Calibrate controller thresholds from the controller's own online beta
    envelope (untreated median -> switch on; stimulated median -> switch off)."""
    base = simulate("none", T=T, seed=seed + 100)       # calibration runs use
    cal_c = simulate("continuous", T=T, seed=seed + 100)  # different noise seeds
    sk = int(1.0 / (5e-3))                              # skip first second
    thr_on = 0.9 * np.median(base["env"][sk:-1])
    thr_off = 1.1 * np.median(cal_c["env"][sk:-1])
    base = simulate("none", T=T, seed=seed)
    cont = simulate("continuous", T=T, seed=seed)
    adap = simulate("adaptive", T=T, seed=seed, thr_on=thr_on, thr_off=thr_off)
    return {"none": base, "continuous": cont, "adaptive": adap}, (thr_on, thr_off)


def main(seeds=(1, 2, 3, 4, 5)):
    rows = {"none": [], "continuous": [], "adaptive": []}
    first = None
    for sd in seeds:
        runs, thr = run_policies(sd)
        first = first or (runs, thr)
        b0 = beta_power_blanked(runs["none"])
        for n, r in runs.items():
            rows[n].append((beta_power_blanked(r) / b0,
                            r["stim"][int(1 * FS):].mean()))
    print(f"{'policy':<12}{'beta power (rel. to none)':>28}{'stim duty (energy proxy)':>28}")
    for n, v in rows.items():
        v = np.array(v)
        print(f"{n:<12}{v[:,0].mean():>18.2f} +/- {v[:,0].std():.2f}"
              f"{v[:,1].mean():>18.2f} +/- {v[:,1].std():.2f}")

    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    first, thr = first
    t = np.arange(len(first["none"]["lfp"])) * DT
    t0 = int(1.0 * FS)
    for j, n in enumerate(["none", "continuous", "adaptive"]):
        r = first[n]
        ax[0, j].plot(t[t0:], r["lfp"][t0:], lw=0.5)
        lo, hi = r["lfp"][t0:].min(), r["lfp"][t0:].max()
        ax[0, j].fill_between(t[t0:], lo, hi, where=r["stim"][t0:], color="orange",
                              alpha=0.25, step="pre")
        ax[0, j].set_title(f"{n} (stim duty {r['stim'][t0:].mean():.0%})")
        ax[0, j].set_xlabel("time (s)")
        te = np.arange(len(r["env"])) * 5e-3
        ax[1, j].plot(te[200:], r["env"][200:], lw=0.8)
        if n == "adaptive":
            ax[1, j].axhline(thr[0], ls="--", c="r", lw=0.8, label="on")
            ax[1, j].axhline(thr[1], ls="--", c="g", lw=0.8, label="off")
            ax[1, j].legend()
        ax[1, j].set_xlabel("time (s)")
        ax[1, j].set_ylim(0, None)
    ax[0, 0].set_ylabel("LFP proxy")
    ax[1, 0].set_ylabel("beta envelope (online)")
    plt.tight_layout()
    plt.savefig("adbs_results.png", dpi=130)


if __name__ == "__main__":
    main()
