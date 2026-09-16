# Optional external spiral data

No external binary is bundled or needed by the synthetic acquisition tests.
If reproducing David's configuration, supply Single spiral.npz here from:
https://github.com/imphys/MRF_undersampling_optimization/blob/4a7c0eddb8be2a6ce8a36c97a01fafcf13e7665e/Data%20structures/Single%20spiral.npz

Provenance: D.G.J. Heesterbeek, imphys/MRF_undersampling_optimization,
revision 4a7c0eddb8be2a6ce8a36c97a01fafcf13e7665e.
Git blob: 2f500c53153b7d46d6b438219235b6f1dfd22715; size 15,252 bytes.
Source repository license: GPL-3.0. Retain attribution/license and clarify
data redistribution terms before bundling this external input.

Schema: Coords, float64, shape (2,1802), rows kx then ky. The reference
interprets these as normalized cycles/pixel with nominal Nyquist radius 0.5;
observed maximum radius 0.4955618988. The archive has no physical FOV, dwell
time, gradient waveform or scanner orientation metadata.

load_spiral_coordinates converts this layout to (samples,2), (ky,kx),
retaining units. It performs no downloading, normalization or clipping.
