from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="gasf-ray-trace",
    version="0.1.0",
    packages=["gasf_ray_trace"],
    ext_modules=[CUDAExtension(
        name="gasf_ray_trace._C",
        sources=["bindings.cpp", "native_trace.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": []},
    )],
    cmdclass={"build_ext": BuildExtension},
)
