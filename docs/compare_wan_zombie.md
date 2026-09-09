# Strategy comparison — scenario `wan_zombie`

grid coverage: 72/72 configurations measured

feasible: 12/72 · Pareto-optimal: 10
best verified mode (grid): `g_c15_b2_r640_on_1000_idx` — critical 15 fps, 
background 2 fps, 640 px, cloud on/1000 ms, indexing on; capability retained 
1.00
  recall 0.982 · p95 1583 ms

Pareto-optimal feasible configurations (capability ↑, CPU ↓, cloud bytes ↓):

| config | capability | edge CPU % | cloud bytes | recall | p95 |
|---|---|---|---|---|---|
| g_c15_b2_r640_on_1000_idx | 1.00 | 185 | 2,260,243 | 0.982 | 1583 ms |
| g_c15_b2_r640_off_idx | 0.82 | 211 | 0 | 0.982 | 451 ms |
| g_c5_b2_r640_on_1000_idx | 0.76 | 92 | 2,229,451 | 0.982 | 1849 ms |
| g_c15_b2_r640_off_noidx | 0.75 | 201 | 0 | 1.000 | 580 ms |
| g_c5_b1_r640_on_1000_idx | 0.74 | 79 | 2,055,582 | 0.965 | 1936 ms |
| g_c15_b1_r640_off_noidx | 0.73 | 179 | 0 | 0.965 | 970 ms |
| g_c5_b1_r640_on_1000_noidx | 0.67 | 77 | 2,055,771 | 0.965 | 1928 ms |
| g_c5_b2_r640_off_idx | 0.58 | 96 | 0 | 0.982 | 588 ms |
| g_c5_b2_r640_off_noidx | 0.51 | 90 | 0 | 0.982 | 583 ms |
| g_c5_b1_r640_off_noidx | 0.49 | 71 | 0 | 0.965 | 919 ms |

| strategy | evaluations to first feasible | capability retained (found) | 
capability retained (best in space) |
|---|---|---|---|
| grid | 72 (exhaustive) | 1.00 | 1.00 |
| random | 5.5 ± 4.7 (p90 12; analytic 5.6; 1000 orders) | 0.73 (mean of first 
found) | 1.00 |
| greedy | 2 (first feasible found) | 1.00 | 1.00 |
| llm:mock | 2 (first feasible found) | 1.00 | 1.00 |

greedy trajectory:
  1. g_c15_b2_r640_on_3000_idx → recall 1.000, p95 5052 ms, FAIL — start at 
highest-capability configuration
  2. g_c15_b2_r640_on_1000_idx → recall 0.982, p95 1583 ms, PASS — latency 
violated with cloud on → cloud_timeout 3000→1000 ms

llm:mock trajectory:
  1. g_c15_b2_r640_on_3000_idx → recall 1.000, p95 5052 ms, FAIL — round 0: 
start from full capability
  2. g_c15_b2_r640_on_1000_idx → recall 0.982, p95 1583 ms, PASS — round 1: 
latency violated with cloud on → cloud_timeout 3000→1000 ms

