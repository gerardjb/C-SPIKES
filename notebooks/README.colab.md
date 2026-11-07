# Colab fluorescence-to-spikes demo

[Open in Colab](https://colab.research.google.com/github/gerardjb/C-SPIKES/blob/review/colab-standalone/notebooks/cspikes_colab.ipynb)

Select a **Python 3.13 GPU runtime**, then **Run all**. No Drive mount, companion Python files or source compilation is required. The notebook creates an isolated environment, installs published `c-spikes==0.2.0rc1` wheels from PyPI and fetches hash-checked recording, parameter and checkpoint assets from this repository. Allow several GB of downloads and about 10 GB of disk space. Free GPU access is not guaranteed.

The example runs GPU PGAS (1,000 particles, 200 iterations, 100 burn-in) and pretrained BiophysML/ENS2 inference on the same first ten seconds of cell01 epoch0. Its supplied timebase has no recorded action potentials in this window. Plots show fluorescence, recorded APs, each model's output on its own scale, and the PGAS log-probability trace. A downloadable ZIP includes results and provenance. Completion does not establish convergence, accuracy or a model ranking.

The selected checkpoint is the owner's preferred December BiophysML/ENS2 model, not a verified best-performing model. Training overlap and checkpoint redistribution terms still need documentation before broad distribution. The processed recording originates from [DANDI 000168](https://dandiarchive.org/dandiset/000168/draft); preserve the upstream CC BY 4.0 attribution. The exact processed-MAT mapping is not yet documented. No model weights are included in exported results.

During review, both the link above and `REPO_REF` in the notebook point to `review/colab-standalone`. The link becomes usable after that branch is pushed. Change both references together after merging. Asset hashes stop the notebook if source content unexpectedly changes. The checkout supplies assets only: the package is imported from the installed wheel.

The notebook stops on missing wheels, invalid assets, unavailable GPU backends or failed inference; it does not silently fall back to CPU or build from source. GPU model/driver names are recorded rather than restricted to a previously tested machine. Native and PyTorch inference use separate fresh processes to release their CUDA contexts, with 20-minute and 5-minute watchdogs respectively. The runtime environment and installation report are saved in the result ZIP. Disconnecting Colab discards temporary files, so download results first.
