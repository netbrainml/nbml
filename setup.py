import setuptools

setuptools.setup(
    name="nbml",
    version="0.0.1",
    author="Dom Huh",
    author_email="dhuh137@gmail.com",
    description="Various tools used for machine learning",
    url="https://github.com/netbrainml/nbml",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.0', 'fastai', 'tensorflow'],

    )