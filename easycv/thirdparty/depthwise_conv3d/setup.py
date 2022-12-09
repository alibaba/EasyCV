import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, sources, includes):
    return CUDAExtension(
        name='{}'.format(name),
        sources=[p for p in sources],
        include_dirs=[i for i in includes],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]})


# -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D_GLIBCXX_USE_CXX11_ABI=1
sources = []
sources.extend(glob.glob('src/*.cu'))
sources.extend(glob.glob('src/*.cpp'))

setup(
    name='depthwise_conv3d',
    version='1.0.3',
    author='gungui98',
    author_email='phi.nguyen.uet@gmail.com',
    url='https://www.github.com',
    description="cuda implementation of 3d depthwise convolution",
    ext_modules=[
        make_cuda_ext(name='DWCONV_CUDA',
                      sources=sources,
                      includes=['src'])
    ],
    py_modules=['depthwise_conv3d'],
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
    ),
    install_requires=['torch>=1.6'],
    keywords=["pytorch", "cuda", "depthwise convolution"],
    cmdclass={'build_ext': BuildExtension}, zip_safe=False)
