import setuptools
from distutils.core import setup
from Cython.Build import cythonize
import numpy

with open("README.md", "r") as file:
    long_desc = file.read()


setup(
    name="OASYS",
    version="0.0.1",
    author="Adam Eck",
    author_email="aeck@oberlin.edu",
    description="Open multiAgent SYStems",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    ext_modules=cythonize(
        [
            #"*.pyx",
            "oasys/agents/*.pyx",
            "oasys/domains/*.pyx",
            "oasys/planning/*.pyx",
            "oasys/structures/*.pyx",
            "oasys/simulation/*.pyx",
            "oasys/domains/wildfire/*.pyx",
        ],
        annotate=True,
        compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()], install_requires=['numpy']
)
