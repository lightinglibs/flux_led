from setuptools import setup

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "pytest-asyncio",
    "ruff==0.11.2",
    "codecov>=2.1.4",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r2>=0.2.7",
    "pytest-runner>=5.2",
    "Sphinx>=3.4.3",
    "sphinx_rtd_theme>=0.5.1",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    "webcolors",
    'typing_extensions;python_version<"3.8"',
    "async_timeout>=3.0.0",
]


extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}


setup(
    name="flux_led",
    packages=["flux_led"],
    version="1.2.0",
    description="A Python library to communicate with the flux_led smart bulbs",
    author="Daniel Hjelseth Høyer",
    author_email="mail@dahoiv.net",
    url="https://github.com/Danielhiversen/flux_led",
    license="LGPL-3.0-or-later",
    include_package_data=True,
    package_data={"flux_led": ["py.typed"]},
    keywords=[
        "flux_led",
        "smart bulbs",
        "light",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: "
        + "GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Home Automation",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    extras_require=extra_requirements,
    entry_points={"console_scripts": ["flux_led = flux_led.fluxled:main"]},
    install_requires=requirements,
)
