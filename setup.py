#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import re
from setuptools import setup, Command

__PATH__ = os.path.abspath(os.path.dirname(__file__))


def read_readme():
    with open('README.md') as f:
        return f.read()


def read_version():
    # importing gpustat causes an ImportError :-)
    with open(os.path.join(__PATH__, 'gpustat/__init__.py')) as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find __version__ string")


__version__ = read_version()


# brought from https://github.com/kennethreitz/setup.py
class DeployCommand(Command):
    description = 'Build and deploy the package to PyPI.'
    user_options = []

    def initialize_options(self): pass
    def finalize_options(self): pass

    @staticmethod
    def status(s):
        print(s)

    def run(self):
        import twine  # we require twine locally  # noqa

        assert 'dev' not in __version__, (
            "Only non-devel versions are allowed. "
            "__version__ == {}".format(__version__))

        with os.popen("git status --short") as fp:
            git_status = fp.read().strip()
            if git_status:
                print("Error: git repository is not clean.\n")
                os.system("git status --short")
                sys.exit(1)

        try:
            from shutil import rmtree
            self.status('Removing previous builds ...')
            rmtree(os.path.join(__PATH__, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution ...')
        os.system('{0} setup.py sdist'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine ...')
        ret = os.system('twine upload dist/*')
        if ret != 0:
            sys.exit(ret)

        self.status('Creating git tags ...')
        os.system('git tag v{0}'.format(__version__))
        os.system('git tag --list')
        sys.exit()


install_requires = [
    'six>=1.7',
    'nvidia-ml-py>=11.450.129,<=11.495.46',  # see #107
    'psutil>=5.6.0',    # GH-1447
    'blessed>=1.17.1',  # GH-126
]

tests_requires = [
    'mockito>=1.2.1',
]
if sys.version_info >= (3, 5):
    tests_requires += ['pytest>=5.4.1', 'pytest-runner']
else:
    tests_requires += ['pytest<5.0', 'more_itertools<8.0', 'attrs<19.2.0']

setup(
    name='gpustat',
    version=__version__,
    license='MIT',
    description='An utility to monitor NVIDIA GPU status and usage',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/wookayin/gpustat',
    author='Jongwook Choi',
    author_email='wookayin@gmail.com',
    keywords='nvidia-smi gpu cuda monitoring gpustat',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: System :: Monitoring',
    ],
    packages=['gpustat'],
    install_requires=install_requires,
    extras_require={'test': tests_requires},
    tests_require=tests_requires,
    entry_points={
        'console_scripts': ['gpustat=gpustat:main'],
    },
    cmdclass={
        'deploy': DeployCommand,
    },
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.4',
)
