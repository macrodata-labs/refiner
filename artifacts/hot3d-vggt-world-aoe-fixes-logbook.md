# HOT3D VGGT-world AoE fixes logbook

Run: `focal-clean-10-vggt-world-aoe-fixes` on clips `clip-001849` through `clip-001858`. Lower score is better.

## Baselines

| experiment | h1 score | trans m | rot deg | jitter m | coverage |
|---|---:|---:|---:|---:|---:|
| `baseline_vggt_world_tw7_raw` | 18.149283 | 0.007170 | 1.923113 | 0.004236 | 0.945302 |
| `baseline_vggt_world_tw7_temporal` | 16.959417 | 0.006286 | 1.893452 | 0.002663 | 0.945302 |

## Best 20

| rank | experiment | h1 score | delta vs temporal baseline | trans m | rot deg | jitter m | coverage |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `window_accel20_jerk0_a0.625_rw9_g0.8_temporal` | 15.909239 | -1.050178 | 0.005308 | 1.889762 | 0.002129 | 0.945302 |
| 2 | `window_accel20_jerk0.05_a0.625_rw9_g0.8_temporal` | 15.909243 | -1.050174 | 0.005308 | 1.889763 | 0.002129 | 0.945302 |
| 3 | `window_accel20_jerk0.1_a0.625_rw9_g0.8_temporal` | 15.909248 | -1.050169 | 0.005308 | 1.889763 | 0.002129 | 0.945302 |
| 4 | `window_accel20_jerk0.5_a0.625_rw9_g0.8_temporal` | 15.909287 | -1.050130 | 0.005308 | 1.889766 | 0.002128 | 0.945302 |
| 5 | `window_accel20_jerk0_a0.675_rw9_g0.8_temporal` | 15.923166 | -1.036251 | 0.005317 | 1.890054 | 0.002158 | 0.945302 |
| 6 | `window_accel20_jerk0.05_a0.675_rw9_g0.8_temporal` | 15.923170 | -1.036247 | 0.005317 | 1.890054 | 0.002158 | 0.945302 |
| 7 | `window_accel20_jerk0.1_a0.675_rw9_g0.8_temporal` | 15.923174 | -1.036243 | 0.005317 | 1.890054 | 0.002158 | 0.945302 |
| 8 | `window_accel20_jerk0.5_a0.675_rw9_g0.8_temporal` | 15.923205 | -1.036212 | 0.005318 | 1.890056 | 0.002158 | 0.945302 |
| 9 | `window_accel20_jerk0_a0.7_rw9_g0.8_temporal` | 15.940105 | -1.019312 | 0.005332 | 1.890238 | 0.002174 | 0.945302 |
| 10 | `window_accel20_jerk0.05_a0.7_rw9_g0.8_temporal` | 15.940108 | -1.019309 | 0.005332 | 1.890238 | 0.002174 | 0.945302 |
| 11 | `window_accel20_jerk0.1_a0.7_rw9_g0.8_temporal` | 15.940111 | -1.019306 | 0.005332 | 1.890239 | 0.002174 | 0.945302 |
| 12 | `window_accel20_jerk0.5_a0.7_rw9_g0.8_temporal` | 15.940136 | -1.019280 | 0.005332 | 1.890241 | 0.002174 | 0.945302 |
| 13 | `window_accel10_jerk0.5_a0.625_rw9_g0.8_temporal` | 15.950864 | -1.008553 | 0.005345 | 1.889220 | 0.002206 | 0.945302 |
| 14 | `window_accel10_jerk0.1_a0.625_rw9_g0.8_temporal` | 15.951796 | -1.007621 | 0.005345 | 1.889232 | 0.002207 | 0.945302 |
| 15 | `window_accel10_jerk0.05_a0.625_rw9_g0.8_temporal` | 15.951918 | -1.007499 | 0.005345 | 1.889234 | 0.002207 | 0.945302 |
| 16 | `window_accel10_jerk0_a0.625_rw9_g0.8_temporal` | 15.952041 | -1.007376 | 0.005346 | 1.889235 | 0.002207 | 0.945302 |
| 17 | `combo_window10_filter_vel35_temporal` | 15.964447 | -0.994970 | 0.005271 | 1.893155 | 0.002211 | 0.944631 |
| 18 | `window_accel10_jerk0.5_a0.675_rw9_g0.8_temporal` | 15.970845 | -0.988572 | 0.005359 | 1.889642 | 0.002240 | 0.945302 |
| 19 | `window_accel10_jerk0.1_a0.675_rw9_g0.8_temporal` | 15.971820 | -0.987597 | 0.005360 | 1.889655 | 0.002242 | 0.945302 |
| 20 | `window_accel10_jerk0.05_a0.675_rw9_g0.8_temporal` | 15.971947 | -0.987470 | 0.005360 | 1.889657 | 0.002242 | 0.945302 |

## Takeaways

- Acceleration-regularized window optimization is the only AoE fix that materially improved this VGGT-world stack.
- Best setting: `window_accel20_jerk0_a0.625_rw9_g0.8_temporal`, score `15.909239`, improving `-1.050178` over `baseline_vggt_world_tw7_temporal`.
- Jerk weight barely matters at `accel=20`; all jerk values tie within numerical noise.
- Depth-rescaled HaWoR translation barely moved the score; best depth variant was only a tiny improvement over baseline.
- Filtering alone did not beat the acceleration-window variants; combined filters also did not beat pure strong acceleration regularization.
