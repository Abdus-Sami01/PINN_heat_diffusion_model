### Forward PINN vs FDM (no data, physics only)

| metric | value |
|---|---|
| relative L2 error | 0.174% |
| final PDE residual | 5.93e-07 |

### Inverse PINN headline (3 sensors, 1 C noise, h0 = 0.15)

| | h |
|---|---|
| true | 0.05 |
| recovered | 0.05161 |
| error | 3.23% |

### h recovery error (%) vs sensors and noise

| sensors | noise (C) | inverse PINN | FDM least-squares |
|---|---|---|---|
| 5 | 0.0 | 0.65% | 0.0% |
| 5 | 1.0 | 3.5% | 1.41% |
| 5 | 2.0 | 6.68% | 2.81% |
| 3 | 0.0 | 0.68% | 0.0% |
| 3 | 1.0 | 2.76% | 0.14% |
| 3 | 2.0 | 1.93% | 0.27% |
| 1 | 0.0 | 32.44% | 0.0% |
| 1 | 1.0 | 62.45% | 7.21% |
| 1 | 2.0 | 55.92% | 13.91% |

### Full-field reconstruction from the same information

| model | uses physics | full grid rel L2 | max abs error (C) |
|---|---|---|---|
| forward PINN | yes | 0.17% | 0.13 |
| data-only NN | no | 20.7% | 20.62 |
| LSTM | no | 37.32% | 54.2 |
