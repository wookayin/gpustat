from setuptools import setup
import sys
import os
import re

IS_PY_2 = (sys.version_info[0] <= 2)


def read_readme():
    with open('README.md') as f:
        return f.read()

def read_version():
    # importing gpustat causes an ImportError :-)
    __PATH__ = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(__PATH__, 'gpustat.py')) as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find __version__ string")


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
    version=read_version(),
    license='MIT',
    description='An utility to monitor NVIDIA GPU status and usage',
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
