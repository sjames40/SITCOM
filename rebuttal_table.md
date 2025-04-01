### Table A: PSNR and NFEs for different methods for the task of super resolution on ImageNet.
|Method |  Avg. PSNR |NFEs |
|---------|------|----------|
| DPS 	| 23.86	|   1000 |
| DAPS 	| 25.67	|  1000 |
| DDNM 	| 23.96	|   1000 |
| RED-diff 	| 24.89|   1000 |
| PGDM 	| 25.22	|   1000  |
| DCDP 	| 24.12	|   400 |
| SITCOM (Ours)	| 26.35	|  $N \times K = 400$ |
### Table B: Comparison of SITCOM and RED-diff for the task of non-linear deblurring on ImageNet.
|Method (Task: NDB) | Avg. PSNR | Avg. SSIM | Avg. LPIPS|
|---------|------|----------|-|
| RED-diff 	| **29.51**	| 0.828| 0.211|
| SITCOM (Ours) 	| 28.78	|**0.832**|**0.16**|
### Table C: **Dataset**: ImageNet. **Metric**: LPIPS. **Noise level**: 0.05. **Source**: Table 2 of MGPS, Table 2 of DCPS, and Table 2 of FPS.
|Method   | SR      | BIP       | RIP         | MDB          | GDB         | NDB         | PR          | HDR         |
|---------|---------|-----------|-------------|--------------|-------------|-------------|-------------|-------------|
| MGPS    | 0.3  | 0.22      | --          | <u>0.22</u>     | <u>0.32</u>   | <u>0.44</u>   | <u>0.47</u>   | **0.1**    |
| DCPS    | <u>0.24</u>  | --      | --          | --     | --   | --   | --   | --    |
| FPS     | 0.33    | **0.2**   | <u>0.32</u> | 0.37         | 0.39        | --          | --          | --          |
| FPS-SMC | 0.31    | <u>0.21</u> | 0.33        | 0.36         | 0.4         | --          | --          | --          |
| SITCOM  | **0.23** | <u>0.21</u> | **0.13**   | **0.18**     | **0.23**    | **0.16**    | **0.24**    | <u>0.16</u> |
- [MGPS] (ICLR25) Variational Diffusion Posterior Sampling with Midpoint Guidance,
- [FPS] (ICLR24) Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective, and
- [DCPS] (NeurIPS24) Divide-and-Conquer Posterior Sampling for Denoising Diffusion priors
### Table D: **Dataset**: FFHQ. **Metric**: PSNR. **Noise level**: 0.01. **Source**: Table 1 and Table 2 of PnP-DM.
|Method |  GDB | MDB | SR | PR |
|---------|------|----------|-|-|
| PnP-DM (VP)     | 29.46    |  30.06 | 29.4 | 30.36 |
| PnP-DM (VE)     | 29.65    | 30.38 | 29.57 | 29.88 |
| PnP-DM (iDDPM)  | 29.6     | 30.26 | 29.53 | 30.61 |
| PnP-DM (EDM)    | 29.66    | 30.35 | 29.6 | 31.14 |
| SITCOM (Ours)   | **32.12**    |  **32.34** | **30.95** | **31.88** |
- [PnP-DM] (NeurIPS24) Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors
### Table E: SITCOM's results using different learning rate $\gamma$. The format is PSNR/run-time (seconds)
|Task |  $\gamma = 0.001$ | $\gamma = 0.005$ | $\gamma = 0.010$ (default) | $\gamma = 0.020$ | $\gamma = 0.050$ |
|---------|------|----------|-|-|-|
| SR 	| 24.02/37.35   |  30.34/35.39 | 30.40/35.20 | 30.64/33.38 | 29.89/30.19 |
| NDB 	| 23.65/38.13   |  30.10/36.34 | 30.27/36.05 | 30.46/32.12 | 29.42/29.45 |
