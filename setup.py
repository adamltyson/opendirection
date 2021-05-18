from setuptools import setup, find_packages

requirements = [
    "numpy",
    # needed until pycircstat is updated
    "scipy <= 1.2.1",
    "pandas",
    "matplotlib",
    "seaborn",
    "pycircstat",
    "nose",
    "decorator",
    "xlrd",
    "imlib >= 0.0.22",
    "packaging",
    "fancylog >= 0.0.8",
    "movement",
    "spikey >= 0.0.5",
    "vedo"
]


setup(
    name="opendirection",
    version="0.1.0",
    description="Analysis of spiking activity in an open field",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "sphinx",
            "recommonmark",
            "sphinx_rtd_theme",
            "pydoc-markdown",
            "black",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coverage>=5.0.3",
        ]
    },
    entry_points={
        "console_scripts": [
            "opendirection = opendirection.main:main",
            "opendirection_batch = opendirection.batch:main",
            "opendirection_multiexp = opendirection.multiexp:main",
            "gen_velo_profile = opendirection.utils.generate_velo_profile:main",
        ]
    },
    url="https://github.com/adamltyson/opendirection",
    author="Adam Tyson",
    author_email="adam.tyson@ucl.ac.uk",
)
