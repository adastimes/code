from setuptools import setup, Extension

# Compile *mysum.cpp* into a shared library
setup(
    #...
    ext_modules=[Extension('yuv2rgb', ['yuv2rgb.cpp'],),],
)
