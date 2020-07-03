from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pytorch_block_sparse',
      version='0.1',
      description='pytorch_block_sparse is a python package for fast block sparse matrices computation.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.0',
        'Topic :: Text Processing',
      ],
      keywords='',
      url='',
      author='',
      author_email='',
      license='MIT',
      packages=['pytorch_block_sparse'],
      install_requires=['click'],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      entry_points={
          'console_scripts': ['pytorch_block_sparse_run=pytorch_block_sparse.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False,
      ext_modules=[
        CUDAExtension('block_sparse_cuda', [
            'pytorch_block_sparse/block_sparse_cuda.cpp',
            'pytorch_block_sparse/block_sparse_cuda_kernel.cu',
        ]),
      ],
      cmdclass={
        'build_ext': BuildExtension
      }
      )

