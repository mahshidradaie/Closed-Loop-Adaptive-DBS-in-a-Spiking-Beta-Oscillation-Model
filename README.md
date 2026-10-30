# Closed-Loop Adaptive Deep Brain Stimulation in a Spiking Beta-Oscillation Model

A delayed excitatory–inhibitory LIF network (400 E / 100 I neurons, pure NumPy/SciPy) that expresses
pathological beta-band (13–30 Hz) synchrony, plus three stimulation policies:

| Policy | Description |
|---|---|
| open-loop | constant-amplitude 130 Hz desynchronising pulse train |
| closed-loop **gated** | online beta-envelope estimate (band-pass of an LFP proxy) with hysteresis thresholds switches stimulation on/off |
| closed-loop **integral** | integral controller regulating beta envelope to a setpoint |

The network's tonic drive alternates between a strong-beta and a weak-beta state (period 2.5 s) to mimic
fluctuating pathology, which is what makes adaptive stimulation worthwhile.

## Run
```
pip install numpy scipy matplotlib
python run_experiment.py      # ~3 min; writes results.json and adbs_results.png
```

## Results (3 seeds, 10 s simulations, `results.json`)
| Policy | Beta power reduction | Stimulation energy (a.u.) | Mean E rate (Hz) |
|---|---|---|---|
| none | – | 0 | 7.6 |
| open-loop (A=4) | 19.5 % | 0.842 | 5.2 |
| **gated closed-loop (A=4)** | 16.1 % | **0.344** | 5.3 |
| integral closed-loop | 6.6 % | 0.932 | 4.8 |

Gated closed-loop stimulation gives most of the beta suppression of open-loop DBS with ~59 % less energy.
The integral controller, as tuned here, did **not** work well (the envelope estimator is noisy and the
regulated variable has little dynamic range); it is kept as a negative result.

## Caveats
Toy model; "stimulation" is an abstract desynchronising current, not a biophysical electrode model. Gains
depend on controller thresholds, which were hand-tuned on a small grid (no held-out tuning). Three seeds only.
