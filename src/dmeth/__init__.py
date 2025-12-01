# main

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dmeth")
except PackageNotFoundError:
    __version__ = "0.2.0.dev"
