"""S9.3 -- ``pyproject.toml``'s version and ``CHANGELOG.md``'s top heading agree.

Two files state the release number, and nothing in the build makes them agree.
That is the whole bug this file is: a changelog announcing a release the package
does not carry, or a package version nobody wrote release notes for.

The format is deliberately narrow. Keep a Changelog wants
``## [2.0.0] - 2026-10-04`` at the top; this asserts exactly that shape, and
reads the version out of the *heading* rather than out of a prose sentence, so
rewording the entry cannot silently desynchronise the two.

``tests/test_pyproject.py`` separately pins the literal version; this file is
what makes the pin meaningful by tying it to the release notes.
"""

from __future__ import annotations

import os
import re

import pytest

from conftest import ROOT_DIR

PYPROJECT_PATH = os.path.join(ROOT_DIR, "pyproject.toml")
CHANGELOG_PATH = os.path.join(ROOT_DIR, "CHANGELOG.md")

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

# `## [2.0.0] - 2026-10-04`, optionally with an "Unreleased" section above it.
# The version group keeps the dot literal so `2.0` is not read as `2` + `0`.
RELEASE_HEADING = re.compile(
    r"^##\s+\[(?P<version>\d+(?:\.\d+)*)\]\s+-\s+(?P<date>\d{4}-\d{2}-\d{2})\s*$"
)


def _pyproject_version() -> str:
    if tomllib is None:  # pragma: no cover - only on a 3.10 interpreter
        pytest.skip("no TOML parser available (need Python >= 3.11 or `tomli`)")
    with open(PYPROJECT_PATH, "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _changelog_headings() -> list[tuple[str, str]]:
    """Every ``## [x.y.z] - date`` heading, in file order."""
    with open(CHANGELOG_PATH, "r", encoding="utf-8") as handle:
        text = handle.read()
    found = []
    for line in text.splitlines():
        match = RELEASE_HEADING.match(line)
        if match:
            found.append((match.group("version"), match.group("date")))
    return found


@pytest.fixture(scope="module")
def pyproject_version() -> str:
    return _pyproject_version()


@pytest.fixture(scope="module")
def headings() -> list[tuple[str, str]]:
    return _changelog_headings()


class TestChangelogShape:
    def test_the_changelog_exists_and_is_not_empty(self) -> None:
        assert os.path.getsize(CHANGELOG_PATH) > 0

    def test_the_top_release_heading_parses(self, headings) -> None:
        assert headings, (
            f"{CHANGELOG_PATH} has no `## [x.y.z] - yyyy-mm-dd` heading, so the "
            "release version cannot be read off it"
        )

    def test_the_top_heading_is_the_newest_release(self, pyproject_version, headings) -> None:
        """The top entry must describe the version the package carries.

        Not merely *an* entry: a 2.0.0 package whose newest changelog entry is
        1.2.1 is a package with unreleased changes in it, which is precisely
        the state this assertion exists to rule out.
        """
        assert headings[0][0] == pyproject_version, (
            f"pyproject.toml ships {pyproject_version} but the newest CHANGELOG "
            f"entry is {headings[0][0]}"
        )

    def test_headings_are_unique(self, headings) -> None:
        versions = [version for version, _ in headings]
        assert len(versions) == len(set(versions)), (
            f"the changelog lists a version twice: {versions}"
        )

    def test_dates_are_iso_formatted(self, headings) -> None:
        for version, date in headings:
            year, month, day = date.split("-")
            assert len(year) == 4 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31, (
                f"{version} has a malformed release date {date!r}"
            )


class TestVersionSync:
    def test_pyproject_version_is_the_release_version(self, pyproject_version) -> None:
        """S9.3 bumped 1.2.1 -> 2.0.6. A structural break, not a feature drop."""
        assert pyproject_version == "2.0.6"

    def test_the_pyproject_and_the_changelog_cannot_drift(self, pyproject_version, headings) -> None:
        """The contract, restated with the failure spelled out.

        If this ever fires after an intentional release, the fix is to add the
        changelog entry -- never to loosen the assertion, and never to bump only
        one of the two files.
        """
        top = headings[0][0]
        assert pyproject_version == top, (
            f"version drift: pyproject.toml says {pyproject_version}, "
            f"CHANGELOG.md says {top}. They must be edited together."
        )

    def test_the_breaking_entry_says_workflows_do_not_break(self) -> None:
        """The reassuring half of the entry, asserted rather than hoped for.

        2.0.0 removes the V1 registration surface, which reads as a breaking
        change to any user with a saved workflow. The one thing that must not
        be quietly dropped from the release notes is the statement that node
        ids and widget order are unchanged.
        """
        with open(CHANGELOG_PATH, "r", encoding="utf-8") as handle:
            text = handle.read()
        assert "Node ids and widget order did not change" in text
        assert "No user workflow breaks" in text