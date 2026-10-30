"""Compare no stimulation / open-loop DBS / closed-loop (integral) aDBS."""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import adbs

CONFIGS = {
    "no stimulation":          dict(policy="none"),
    "open-loop (A=4)":         dict(policy="open", dbs_amp=4.0),
    "closed-loop gated (A=4)": dict(policy="gated", dbs_amp=4.0, thr_on=0.72, thr_off=0.66),
    "closed-loop integral":    dict(policy="closed", setpoint=0.60, ki=3e-3),
}
SEEDS = [1, 2, 3]
res = {}
traces = {}
for name, cfg in CONFIGS.items():
    rows = []
    for sd in SEEDS:
        r = adbs.simulate(seed=sd, **cfg)
        bp = adbs.beta_power(r["lfp"])[0]
        energy = float((r["amp"] ** 2 * r["stim"]).mean())
        rate = float(r["rate"].sum() / adbs.NE / (adbs.T_MS / 1000))
        rows.append((bp, energy, rate))
        if sd == SEEDS[0]: traces[name] = r
    a = np.array(rows)
    res[name] = dict(beta_power_mean=a[:, 0].mean(), beta_power_sd=a[:, 0].std(),
                     stim_energy=a[:, 1].mean(), mean_rate_hz=a[:, 2].mean())
base = res["no stimulation"]["beta_power_mean"]
for k, v in res.items():
    v["beta_reduction_pct"] = 100 * (1 - v["beta_power_mean"] / base)
    v["reduction_per_energy"] = (v["beta_reduction_pct"] / v["stim_energy"]) if v["stim_energy"] > 0 else None
json.dump(res, open("results.json", "w"), indent=2)
print(f"{'policy':26s} beta(a.u.)  reduction%  energy  rate(Hz)  red%/energy")
for k, v in res.items():
    pe = "-" if v["reduction_per_energy"] is None else f"{v['reduction_per_energy']:.1f}"
    print(f"{k:26s} {v['beta_power_mean']:.3f}      {v['beta_reduction_pct']:6.1f}     {v['stim_energy']:.3f}   {v['mean_rate_hz']:.1f}      {pe}")

fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
t = np.arange(len(traces["no stimulation"]["lfp"])) * adbs.DT / 1000
for k, c in zip(CONFIGS, ["gray", "tab:orange", "tab:blue", "tab:green"]):
    ax[0].plot(t[::50], traces[k]["env"][::50], label=k, color=c, lw=1)
ax[0].set_ylabel("beta envelope (a.u.)"); ax[0].legend(fontsize=7, ncol=2)
ax[1].plot(t[::50], traces["closed-loop gated (A=4)"]["amp"][::50], color="tab:blue")
ax[1].set_ylabel("closed-loop stim amp")
ax[2].plot(t[::50], [adbs.drive_schedule(x * 1000) for x in t[::50]], color="k")
ax[2].set_ylabel("tonic drive (state)"); ax[2].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig("adbs_results.png", dpi=150)
