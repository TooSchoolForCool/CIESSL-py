# python setup.py bdist_wheel
# sudo pip install ./dist/*.whl

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    README = fh.read()

requirements = [
    # 'pyaudio' avoid travis-ci build error
    'webrtcvad'
    'scipy'
    'numpy'
    'samplerate'
]

setup_requirements = [
    # TODO: put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest'
]

setup(
    name="ciessl",
    version="0.0.1",

    author="Zeyu Zhang",
    author_email="zeyuz@outlook.com",
    
    description="Sound Source Localization in Complex Indoor Environment",
    long_description=README,
    
    url="https://github.com/TooSchoolForCool/CIESSL-py",
    
    packages=find_packages(),
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