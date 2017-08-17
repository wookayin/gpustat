from setuptools import setup
import sys

IS_PY_2 = (sys.version_info[0] <= 2)

import gpustat

def read_readme():
    with open('README.md') as f:
        return f.read()

install_requires = [
    'six',
    'nvidia-ml-py>=7.352.0' if IS_PY_2 else \
        'nvidia-ml-py3>=7.352.0',
    'psutil',
    'blessings>=1.6',
]

tests_requires = [
    'mock>=2.0.0',
    'nose',
    'nose-cover3'
]

setup(
    name='gpustat',
    version=gpustat.__version__,
    license='MIT',
    description='An utility to monitor NVIDIA GPU status (wrapper of nvidia-smi)',
    long_description=read_readme(),
    url='https://github.com/wookayin/gpustat',
    author='Jongwook Choi',
    author_email='wookayin@gmail.com',
    keywords='nvidia-smi gpu cuda monitoring gpustat',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: System :: Monitoring',
    ],
    #packages=['gpustat'],
    py_modules=['gpustat'],
    install_requires=install_requires,
    extras_require={'test': tests_requires},
    tests_require=tests_requires,
    test_suite='nose.collector',
    entry_points={
        'console_scripts': ['gpustat=gpustat:main'],
    },
    include_package_data=True,
    zip_safe=False,
)
