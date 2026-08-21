# Platform package sources

`macos/README.md` and `jetson/README.md` are copied to the root of their
respective archive by `scripts/build_platform_packages.py`. The builder always
uses current tracked source files and excludes datasets, model files, engines,
virtual environments, and result records.
