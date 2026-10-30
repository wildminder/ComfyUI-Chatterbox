"""S3.3/S3.4/S3.5 -- packaging.

Two failure modes this file exists to prevent:

* **Duplication.** ``requirements.txt`` and ``pyproject.toml`` disagree, and
  the version nobody reads is the one ComfyUI Manager installs. The current
  repository already ships the ``tqdm`` version of that bug.
* **Re-declaring the host environment.** Naming ``torch`` in a custom node's
  requirements makes ComfyUI Manager try to reconcile it with the user's build.
  ComfyUI bundles the ML stack; this node must declare only what it adds.

The deliberate absence of version pins is asserted too. A pin here downgrades a
package the user already has, which is exactly the fight we are avoiding.
"""

from __future__ import annotations

import os
import re

import pytest

from conftest import ROOT_DIR, COMFYUI_ROOT

REQUIREMENTS_PATH = os.path.join(ROOT_DIR, "requirements.txt")
PYPROJECT_PATH = os.path.join(ROOT_DIR, "pyproject.toml")

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

# What the ComfyUI environment already provides. Declaring any of these is the
# mistake; see the module docstring.
COMFYUI_BUNDLED: frozenset[str] = frozenset(
    {
        "torch",
        "torchaudio",
        "transformers",
        "huggingface_hub",
        "numpy",
        "librosa",
        "soundfile",
        "safetensors",
        "tokenizers",
        "einops",
        "scipy",
        "tqdm",
        "accelerate",
        "diffusers",
    }
)

# Everything ComfyUI must install for this node to work at all.
REQUIRED_DEPS: frozenset[str] = frozenset({"s3tokenizer"})

OPTIONAL_EXTRAS: dict[str, frozenset[str]] = {
    "watermarking": frozenset({"resemble-perth"}),
    "multilingual": frozenset({"pykakasi", "spacy-pkuseg"}),
    "turbo": frozenset({"pyloudnorm"}),
}

# A requirement is a bare distribution name. Anything with a version operator,
# an environment marker or a URL is a pin or a lock, and both are banned.
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def read_requirements() -> list[str]:
    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        ]


def read_pyproject() -> dict:
    if tomllib is None:  # pragma: no cover - only on a 3.10 interpreter
        pytest.skip("no TOML parser available (need Python >= 3.11 or `tomli`)")
    with open(PYPROJECT_PATH, "rb") as handle:
        return tomllib.load(handle)


def project(pyproject: dict) -> dict:
    return pyproject["project"]


def distribution_name(requirement: str) -> str:
    return requirement.split("[", 1)[0].strip()


@pytest.fixture(scope="module")
def requirements() -> list[str]:
    return read_requirements()


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return read_pyproject()


def version_tuple(version: str) -> tuple[int, ...]:
    """Numeric prefix of a version, e.g. ``0.37.0.dev1`` -> ``(0, 37, 0)``."""
    numbers = re.match(r"^(\d+(?:\.\d+)*)", version)
    assert numbers, f"cannot interpret version {version!r}"
    return tuple(int(part) for part in numbers.group(1).split("."))


def satisfies(actual: str, specifier: str) -> bool:
    """Evaluate a single-comparator PEP 440-style specifier.

    Only what this repository actually declares is supported: ``>=`` and ``>``
    with a dotted numeric version. Anything else is a fail, not a silent pass.
    """
    match = re.fullmatch(r"\s*(>=|>|<=|<|==|!=)\s*([\w.+!-]+)\s*", specifier)
    assert match, f"unsupported version specifier {specifier!r}"
    operator, wanted = match.groups()
    left, right = version_tuple(actual), version_tuple(wanted)
    length = max(len(left), len(right))
    left += (0,) * (length - len(left))
    right += (0,) * (length - len(right))
    return {
        ">=": left >= right,
        ">": left > right,
        "<=": left <= right,
        "<": left < right,
        "==": left == right,
        "!=": left != right,
    }[operator]


class TestRequirementsPyprojectParity:
    """``requirements.txt`` and ``pyproject.toml`` must tell one story."""

    def test_requirements_are_unpinned_bare_names(self, requirements: list[str]) -> None:
        for requirement in requirements:
            assert _REQUIREMENT_NAME.match(requirement), (
                f"{requirement!r} is not a bare distribution name -- requirements "
                f"must not carry version pins, markers or URLs"
            )

    def test_pyproject_dependencies_are_unpinned_bare_names(self, pyproject: dict) -> None:
        for requirement in project(pyproject)["dependencies"]:
            assert _REQUIREMENT_NAME.match(distribution_name(requirement)), (
                f"{requirement!r} pins or qualifies a dependency"
            )

    def test_extras_are_unpinned_bare_names(self, pyproject: dict) -> None:
        for extra, packages in project(pyproject).get("optional-dependencies", {}).items():
            for requirement in packages:
                assert _REQUIREMENT_NAME.match(distribution_name(requirement)), (
                    f"extra {extra!r} pins {requirement!r}"
                )

    def test_no_version_pins_anywhere(self, pyproject: dict, requirements: list[str]) -> None:
        everything = list(requirements)
        everything += list(project(pyproject)["dependencies"])
        for packages in project(pyproject).get("optional-dependencies", {}).values():
            everything += list(packages)
        pins = [item for item in everything if not _REQUIREMENT_NAME.match(distribution_name(item))]
        assert not pins, f"version pins found: {pins}"

    def test_requirements_are_a_subset_of_pyproject(self, pyproject: dict, requirements: list[str]) -> None:
        declared = {distribution_name(d) for d in project(pyproject)["dependencies"]}
        missing = sorted(set(requirements) - declared)
        assert not missing, (
            f"{missing} are installed from requirements.txt but not declared in "
            f"pyproject.toml; the two must agree or the registry build differs "
            f"from the local install"
        )

    def test_every_hard_dependency_is_declared(self, pyproject: dict) -> None:
        declared = {distribution_name(d) for d in project(pyproject)["dependencies"]}
        assert REQUIRED_DEPS <= declared, f"missing hard dependencies: {REQUIRED_DEPS - declared}"

    def test_no_comfyui_bundled_package_is_required(self, requirements: list[str]) -> None:
        bundled = sorted(set(requirements) & COMFYUI_BUNDLED)
        assert not bundled, (
            f"{bundled} ship with ComfyUI; re-declaring them makes ComfyUI "
            f"Manager fight the user's torch build"
        )

    def test_no_comfyui_bundled_package_is_declared(self, pyproject: dict) -> None:
        declared = {distribution_name(d) for d in project(pyproject)["dependencies"]}
        bundled = sorted(declared & COMFYUI_BUNDLED)
        assert not bundled, f"{bundled} ship with ComfyUI and must not be declared"

    @pytest.mark.parametrize("extra", sorted(OPTIONAL_EXTRAS))
    def test_extra_exists_with_the_documented_packages(self, pyproject: dict, extra: str) -> None:
        packages = project(pyproject).get("optional-dependencies", {}).get(extra)
        assert packages is not None, f"the {extra!r} extra is missing"
        assert {distribution_name(p) for p in packages} == OPTIONAL_EXTRAS[extra]

    def test_optional_packages_are_not_hard_requirements(self, pyproject: dict, requirements: list[str]) -> None:
        optional = set().union(*OPTIONAL_EXTRAS.values())
        hard = set(requirements) | {distribution_name(d) for d in project(pyproject)["dependencies"]}
        assert not optional & hard, (
            f"{sorted(optional & hard)} are optional extras and must not be "
            f"hard requirements -- the node works without them"
        )

    def test_diffusers_is_in_neither_list(self, pyproject: dict, requirements: list[str]) -> None:
        """diffusers 0.27 cannot import under huggingface_hub 1.x, and the
        vendored tree no longer imports it. Re-adding it re-adds the breakage."""
        assert "diffusers" not in requirements
        declared = {distribution_name(d) for d in project(pyproject)["dependencies"]}
        assert "diffusers" not in declared

    def test_conformer_and_omegaconf_are_in_neither_list(self, pyproject: dict, requirements: list[str]) -> None:
        """Both were dead imports in the vendored tree and were dropped on
        2026-10-07. Re-adding them re-adds dead code."""
        declared = {distribution_name(d) for d in project(pyproject)["dependencies"]}
        for package in ("conformer", "omegaconf"):
            assert package not in requirements
            assert package not in declared

    def test_the_old_tqdm_parity_bug_is_gone(self, pyproject: dict) -> None:
        """It used to be declared in pyproject but missing from requirements."""
        declared = {distribution_name(d) for d in project(pyproject)["dependencies"]}
        assert "tqdm" not in declared
        assert "tqdm" not in read_requirements()


class TestPyprojectMetadata:
    def test_pyproject_parses(self, pyproject: dict) -> None:
        assert project(pyproject)["name"]

    def test_name_is_the_registered_name(self, pyproject: dict) -> None:
        """The registry entry is immutable: do not rename the package.

        The node is registered on comfyregistry as ``ComfyUI-ChatterboxTTS``;
        changing the ``name`` breaks the registry identity (and
        ComfyUI-Manager rewrites the file back anyway). If this test fails,
        revert the edit -- the name is not ours to change.
        """
        assert project(pyproject)["name"] == "ComfyUI-ChatterboxTTS"

    def test_registry_identity_is_present(self, pyproject: dict) -> None:
        comfy = pyproject["tool"]["comfy"]
        assert comfy["PublisherId"].strip()
        assert comfy["DisplayName"].strip()

    def test_requires_python_is_declared(self, pyproject: dict) -> None:
        specifier = project(pyproject)["requires-python"]
        assert satisfies("3.10", specifier)
        assert satisfies("3.13", specifier)

    def test_requires_comfyui_is_declared(self, pyproject: dict) -> None:
        assert "requires-comfyui" in pyproject["tool"]["comfy"]

    def test_requires_comfyui_is_satisfied_by_the_running_core(self, pyproject: dict) -> None:
        specifier = pyproject["tool"]["comfy"]["requires-comfyui"]
        core_version = _running_comfyui_version()
        assert satisfies(core_version, specifier), (
            f"the running ComfyUI core is {core_version}, which does not satisfy "
            f"requires-comfyui = {specifier!r}"
        )

    def test_classifiers_are_present(self, pyproject: dict) -> None:
        classifiers = project(pyproject).get("classifiers", [])
        assert "Programming Language :: Python :: 3" in classifiers
        assert any(c.startswith("Development Status ::") for c in classifiers)

    def test_version_is_the_release_version(self, pyproject: dict) -> None:
        """S9.3 landed the release bump; the cross-file sync lives in
        ``tests/test_version_sync.py``, which also compares against the
        changelog's top heading."""
        assert project(pyproject)["version"] == "2.0.6"

    def test_repository_url_is_preserved(self, pyproject: dict) -> None:
        assert project(pyproject)["urls"]["Repository"].endswith("ComfyUI-Chatterbox")


def _running_comfyui_version() -> str:
    """Version of the ComfyUI checkout the suite is pointed at."""
    path = os.path.join(COMFYUI_ROOT, "comfyui_version.py")
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    match = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", source)
    assert match, f"cannot read __version__ from {path}"
    return match.group(1)