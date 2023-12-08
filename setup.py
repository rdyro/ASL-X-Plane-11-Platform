from setuptools import setup, find_packages
from pathlib import Path
from glob import glob
import toml

pyproject_toml = toml.loads(Path("pyproject.toml").read_text())
name = pyproject_toml["project"]["name"]
version = pyproject_toml["project"]["version"]

package_data = [
    str(Path(__file__).absolute().parent / "aslxplane" / "data" / f)
    for f in ["dynamics_linear.json"]
]

setup(
    name=name,
    version=version,
    python_requires=">=3.7",
    packages=find_packages(),
    description="TODO",
    package_data={"aslxplane": package_data},
    url="https://github.com/nasa/XPlaneConnect/",
)
