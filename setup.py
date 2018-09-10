# python setup.py bdist_wheel
# sudo pip install ./dist/*.whl

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    README = fh.read()

requirements = [
    'pyaudio',
    'webrtcvad',
    'scipy',
    'numpy',
    'samplerate',
    'enum34',
    'matplotlib',
    'scikit-learn',
    'opencv-python',
    'librosa',
    'future',
    'scikit-multilearn',
    'seaborn',
    'pandas'
]

setup_requirements = [
    # TODO: put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest'
]

setup(
    name="ciessl_py_pkgs",
    version="1.0.0",

    author="Zeyu Zhang",
    author_email="zeyuz@outlook.com",
    
    description="Sound Source Localization in Complex Indoor Environment",
    long_description=README,
    
    url="https://github.com/TooSchoolForCool/CIESSL-py",
    
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,

    license="Apache-2.0",

    test_suite='tests',

    install_requires=requirements,
    
    tests_require=test_requirements,

    setup_requires=setup_requirements,

    classifiers=(
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ),
)