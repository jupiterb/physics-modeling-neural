from setuptools import setup, find_packages

requirements = ["numpy", "pydantic", "scipy", "tables", "torch"]

setup(
    name="Physics Learn",
    version="0.0.0",
    description="Physics learn with PINNs and more",
    packages=find_packages(),
    install_requires=requirements,
)
