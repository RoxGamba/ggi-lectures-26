# GWs from compact binaries 101

Slides, toy models and animations for two lectures on waveform modelling given at the
Galileo Galilei Institute (Florence), 2026.

The structure of some of these scripts is in part based on other
libraries from S.~Bernuzzi, S.~Albanesi and D.~Chiaramello. I warmly thank them
for their continuous collaboration and support.

## Running the toy models

Needs `numpy`, `scipy`, `sympy`, `matplotlib`.

```bash
cd scripts/toy_kerr
python example.py        # inspiral and plunge on Kerr
python animations.py     # the movies (needs ffmpeg)

cd scripts/data
tar -xf toy_surrogate_data.tar && tar -xf toy_phenom_data.tar   # the trning waveforms
cd ..
python toy_surrogate.py  # builds the surrogate from the waveforms in data/
python toy_phenom.py     # builds the mock Phenom model from the waveforms in data/
```

The mock Phenom and surrogate models are trained on [TEOBResumS](https://github.com/teobresums/teobresums)
waveforms, all included in `scripts/data/` (two tarballs). To regenerate them (needs TEOBResumS):
`python toy_phenom.py data`, `python toy_surrogate.py data`.

Challenge: do the same, do better, and extend (spins, higher modes, eccentricity...).

## Licence

* Code (`scripts/`): GPL-3.0, see `LICENSE`. Copyright (C) 2026 Rossella Gamba
* Slides and movies (`slides/`, `movies/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  This covers the original material only. Figures reproduced from published papers are credited on
  each slide and remain under the copyright of their authors and publishers.
