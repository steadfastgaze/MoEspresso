"""MoEspresso: manifest-driven MoE inference on Apple Silicon.

Phases (artifact-centered):
    inventory -> probe -> optimize -> package -> runtime

Each phase consumes and produces durable, versioned, content-hashed artifacts.
The public runtime serves explicit packages for DeepSeek-V4-Flash and Ornith
1.0 35B, resident or through bounded routed-expert residency.
"""

from importlib import metadata as importlib_metadata

try:
    __version__ = importlib_metadata.version("moespresso")
except importlib_metadata.PackageNotFoundError:
    __version__ = "0+unknown"
