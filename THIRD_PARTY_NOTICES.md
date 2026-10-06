# Third-party notices

GASF-GS retains the license and attribution of the source components it uses.

| Component | Location | Attribution / license |
| --- | --- | --- |
| Gaussian Splatting utilities and Gaussian optimization | `gasf/model.py`, `gasf/utils/`, `gasf/losses.py` | Inria / GRAPHDECO, 2023; Gaussian Splatting research and evaluation license in `LICENSE` |
| Spherical harmonic utilities | `gasf/utils/sh.py` | PlenOctree authors, 2021; BSD-style terms retained in the source header |
| Native Gaussian rasterizer and its fisheye extensions | `third_party/fisheye_rasterizer/` | Original Inria / GRAPHDECO notices retained; `LICENSE.md` |
| CUDA nearest-neighbor initialization | `third_party/simple_knn/` | Inria / GRAPHDECO notices retained; `LICENSE.md` |
| Native ray trace/replay integration | `third_party/ray_trace/` | Uses the Gaussian Splatting renderer's data layout and behavior; inherited terms in `LICENSE.md` |
| GLM headers | `third_party/fisheye_rasterizer/third_party/glm/` | G-Truc Creation / GLM authors; original `copying.txt` and header notices retained |

The `simple_knn` extension supplies the original nearest-neighbor scale
initialization. It does not implement the retired KNN regularization experiment.

Please cite the upstream 3D Gaussian Splatting, unscented Gaussian projection,
COLMAP, and dataset publications appropriate to your use. This repository does
not distribute FisheyeNeRF or ScanNet++ data, LPIPS/VGG pretrained weights,
or access credentials.
