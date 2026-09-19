import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jaxfss",
    version="0.2.0",
    install_requires=[
        "jax>=0.4.30,<0.11.0",
        "jaxlib>=0.4.30,<0.11.0",
        "flax>=0.8.4",
        "optax>=0.1.5",
        "distrax>=0.1.3",
        "softclip>=0.1.0",
        "numpy",
    ],
    author="Ryosuke Yoneda",
    author_email="13e.e.c.13@gmail.com",
    description="JAX/Flax NNX implementation of finite-size scaling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yonesuke/jaxfss",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
