from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
rootdir = os.path.dirname(os.path.realpath(__file__))

version = "0.1.2"

setup(name='pytorch_block_sparse',
      version=version,
      description='PyTorch extension for fast block sparse matrices computation, drop in replacement for torch.nn.Linear.',
      long_description="pytorch_block_sparse is a PyTorch extension for fast block sparse matrices computation, drop in replacement for torch.nn.Linear",
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.0',
      ],
      keywords='PyTorch,sparse,matrices,machine learning',
      url='https://github.com/huggingface/pytorch_block_sparse',
      author='François Lagunas',
      author_email='francois.lagunas@m4x.org',
      download_url=f'https://test.pypi.org/project/pytorch-block-sparse/{version}/',
      license='BSD 3-Clause "New" or "Revised" License',
      packages=['pytorch_block_sparse'],
      install_requires=[],
      include_package_data=True,
      zip_safe=False,
      ext_modules=[
        CUDAExtension('block_sparse_native',
                      ['pytorch_block_sparse/native/block_sparse_native.cpp',
                      'pytorch_block_sparse/native/block_sparse_cutlass_kernel_back.cu',
                      'pytorch_block_sparse/native/block_sparse_cutlass_kernel.cu'],
                      extra_compile_args=['-I', '%s/pytorch_block_sparse' % rootdir]
                      ),
      ],
      cmdclass={
        'build_ext': BuildExtension
      }
      )

