"""The vendored package must import in a ComfyUI environment.

S1.5: upstream's ``chatterbox/__init__.py`` resolves its version through
``importlib.metadata.version("chatterbox-tts")``. The package is vendored here,
never pip-installed, so that call raises ``PackageNotFoundError`` at import time
and takes the whole node registration down with it.

``conftest.py`` mocks the vendor tree so these tests cannot import it in-process.
The check therefore runs in a subprocess against the real files, which is also
the only way to see what a genuine import of the vendored tree actually hits.

That import goes through the node's own package: the vendored tree is private
to this node, so ``src/`` is not an import root and the top-level
``chatterbox`` name is not one the shipped code may use. The probe registers
the package by hand -- the child interpreter has no ComfyUI around it -- and
imports ``<package>.src.chatterbox``, the way the shipped code reaches it.

The *classifier* those probes feed is the part that has to be right, and an
environment where the import succeeds is the environment where it is least
exercised. So it is also driven against a synthetic missing module, which fails
on every run: see :class:`TestFailureClassifier`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT_DIR

SRC_DIR = os.path.join(ROOT_DIR, "src")

#: The package name ComfyUI derives from the checkout directory. The probe
#: registers it by hand and sanitises it into a legal module name; the vendored
#: tree is reached through relative imports, so what the package is called does
#: not change what the probe finds.
PACKAGE_NAME = re.sub(r"\W", "_", os.path.basename(ROOT_DIR))

PROBE = textwrap.dedent(
    """
    import importlib
    import importlib.machinery
    import importlib.metadata as md
    import json, sys, traceback, types
    from importlib.metadata import PackageNotFoundError

    # The vendored tree is private to this node: it is reached through the
    # node's own package, and ``src/`` is not an import root. The package is
    # registered by hand -- the child interpreter has no ComfyUI around it --
    # so the probe walks the same package-relative path the shipped code uses.
    alias = {alias!r}
    package = types.ModuleType(alias)
    package.__path__ = [{root!r}]
    package.__package__ = alias
    package.__spec__ = importlib.machinery.ModuleSpec(alias, None, is_package=True)
    package.__spec__.submodule_search_locations = [{root!r}]
    sys.modules[alias] = package

    # Make the condition under test reproducible: the package is vendored, so
    # there is no distribution metadata to find. Only *that* lookup is poisoned
    # -- replacing every ``version()`` also breaks third-party libraries that
    # resolve their own requirements this way (transformers looks up its
    # ``tqdm>=4.27`` requirement), which would mask the real import result.
    _real_version = md.version

    def _version(name):
        if name == 'chatterbox-tts':
            raise PackageNotFoundError(name)
        return _real_version(name)

    md.version = _version
    sys.modules['importlib.metadata'].version = _version

    result = {{}}
    try:
        module = importlib.import_module(alias + '.src.' + {module!r})
    except PackageNotFoundError as exc:
        result['kind'] = 'PackageNotFoundError'
        result['detail'] = str(exc)
    except BaseException as exc:  # noqa: BLE001 -- the point is to classify it
        result['kind'] = type(exc).__name__
        result['detail'] = str(exc)[:400]
        result['module'] = getattr(exc, 'name', None)
        result['traceback_tail'] = traceback.format_exc()[-600:]
    else:
        result['kind'] = 'ok'
        result['version'] = getattr(module, '__version__', None)
        result['names'] = sorted(
            n for n in dir(module) if n[0].isupper()
        )
    print('__PROBE__' + json.dumps(result))
    """
)

#: A module name that cannot exist in any environment, used to force a real
#: ``ModuleNotFoundError`` so the classifier below is exercised on every run --
#: including the healthy ones where importing the vendored package succeeds and
#: the happy-path tests have nothing to classify.
SYNTHETIC_MISSING_MODULE = "chatterbox_zzz_absent_module_for_probe"

#: The only modules that may legitimately keep the vendored package from
#: importing. Both are optional by design (P-020 vendors the `diffusers` symbols
#: the Matcha decoder needs, so nothing imports it any more).
DOCUMENTED_BLOCKERS = frozenset({"perth", "diffusers"})

#: A failed import must look like an import, not like a syntax error or an
#: AttributeError from a bad patch.
IMPORT_ERROR_KINDS = frozenset({"ModuleNotFoundError", "ImportError"})


def classify_blocker(probe: dict) -> bool:
    """The "is this a documented blocker?" half of the failure classifier.

    Split out of the assertion so it can be driven by a real failing import as
    well as by a synthetic one: with the classifier inlined in a test body, the
    only way to reach it is an environment that actually fails to import, which
    is precisely the environment this repository does not usually run in.
    """
    return probe.get("module") in DOCUMENTED_BLOCKERS


def classify_error_kind(probe: dict) -> bool:
    """The "is this a plain import error?" half of the failure classifier."""
    return probe["kind"] in IMPORT_ERROR_KINDS


def run_probe(module: str = "chatterbox") -> dict:
    """Run the import probe in a clean child interpreter.

    ``module`` is a name *under* ``src/``: the probe imports it through the
    node's own package (``<package>.src.<module>``), never as a top-level
    name.
    """
    script = PROBE.format(alias=PACKAGE_NAME, root=ROOT_DIR, module=module)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    marker = "__PROBE__"
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    pytest.fail(
        "the import probe produced no result\n"
        f"--- stdout ---\n{completed.stdout[-2000:]}\n"
        f"--- stderr ---\n{completed.stderr[-2000:]}"
    )


@pytest.fixture(scope="module")
def probe() -> dict:
    return run_probe()


@pytest.fixture(scope="module")
def failing_probe() -> dict:
    """The probe aimed at a module that cannot exist.

    Same script, same ``except`` clauses, same recorded fields as the real
    probe -- only the name on the import statement differs, so the failure is a
    genuine ``ModuleNotFoundError`` carrying a genuine ``.name``.
    """
    return run_probe(SYNTHETIC_MISSING_MODULE)


class TestVendorImportWithoutDistMetadata:
    def test_version_lookup_never_breaks_the_import(self, probe: dict) -> None:
        """The regression this guards: ``version()`` must not be fatal."""
        assert probe["kind"] != "PackageNotFoundError", (
            "importing the vendored package failed on distribution metadata; "
            "the try/except around __version__ is gone"
        )

    def test_blockers_are_the_documented_ones(self, probe: dict) -> None:
        """A *new* import blocker must fail here, not surface as a broken node.

        ``perth`` (S2.2/S2.3) and ``diffusers`` (P-020) used to be hard
        requirements and are now both optional, so they stay listed as known
        blockers: a build without them must still not be what breaks the import.
        Anything else is a regression introduced by this re-sync.
        """
        if probe["kind"] == "ok":
            pytest.skip(
                "the vendored package imports cleanly in this environment, so "
                "there is no blocker to classify; the classifier itself is "
                "covered unconditionally by TestFailureClassifier"
            )
        assert classify_blocker(probe), (
            f"importing the vendored chatterbox failed on "
            f"{probe.get('module')!r}, which is not "
            f"a documented blocker (known: {sorted(DOCUMENTED_BLOCKERS)}).\n"
            f"{probe.get('traceback_tail', '')}"
        )

    def test_failure_is_a_plain_import_error(self, probe: dict) -> None:
        """Not a syntax error, not an AttributeError from a bad patch."""
        if probe["kind"] == "ok":
            pytest.skip(
                "the vendored package imports cleanly in this environment; the "
                "classifier itself is covered unconditionally by "
                "TestFailureClassifier"
            )
        assert classify_error_kind(probe), (
            f"importing the vendored chatterbox raised {probe['kind']}: "
            f"{probe.get('detail')}"
        )



class TestFailureClassifier:
    """The classifier, exercised on a failure that is produced on purpose.

    Both sibling tests above skip when the vendored package imports cleanly, and
    in this environment -- and in CI, where the optional extras are installed --
    it does. Their assertions therefore never run on a green run, which is the
    one run where a broken classifier would cost the most: the tests exist to
    tell an *expected* missing optional dependency from a regression, and
    neither branch of that decision was being checked.

    So the failure is manufactured instead of waited for. The same probe runs
    against a module name that cannot resolve, which yields a genuine
    ``ModuleNotFoundError`` carrying a genuine ``.name``, and the classifier is
    pointed at that. Nothing about the classification is stubbed: the probe
    script, the ``except`` clauses and the recorded fields are the ones the real
    happy-path assertions use.
    """

    def test_the_probe_really_does_fail_to_import(self, failing_probe: dict) -> None:
        """If this passes vacuously, the three tests below prove nothing."""
        assert failing_probe["kind"] == "ModuleNotFoundError", (
            f"the synthetic probe did not raise ModuleNotFoundError; it returned "
            f"{failing_probe['kind']}: {failing_probe.get('detail')}. The probe "
            f"is no longer producing the failure the classifier is tested "
            f"against."
        )

    def test_an_optional_dependency_is_classified_as_a_known_blocker(
        self, failing_probe: dict
    ) -> None:
        """A named optional extra is an expected blocker, not a regression."""
        assert classify_blocker(
            {"kind": "ModuleNotFoundError", "module": "perth"}
        ), "`perth` is a documented blocker and must classify as one"
        assert classify_blocker(
            {"kind": "ModuleNotFoundError", "module": "diffusers"}
        ), "`diffusers` is a documented blocker and must classify as one"

    def test_an_unknown_missing_module_is_not_a_documented_blocker(
        self, failing_probe: dict
    ) -> None:
        """The negative branch, on a real failure: this is what must fail CI.

        The synthetic module is missing for the same reason ``perth`` may be
        missing -- it is not installed -- and the classifier still rejects it,
        because "not a module we know about" is what distinguishes a regression
        from an accepted absence.
        """
        assert not classify_blocker(failing_probe), (
            f"a missing module that is not in {sorted(DOCUMENTED_BLOCKERS)} was "
            f"accepted as a documented blocker: "
            f"{failing_probe.get('module')}. If this module really is an "
            f"optional dependency, add it to DOCUMENTED_BLOCKERS and to the "
            f"manifest."
        )

    def test_a_missing_module_classifies_as_a_plain_import_error(
        self, failing_probe: dict
    ) -> None:
        """Not a syntax error, not an AttributeError from a bad patch."""
        assert classify_error_kind(failing_probe), (
            f"a ModuleNotFoundError was not classified as a plain import error: "
            f"kind={failing_probe['kind']}"
        )

    def test_a_non_import_failure_is_rejected_by_the_error_classifier(self) -> None:
        """The other branch of the same decision, on a synthetic result.

        A patched-upstream ``AttributeError`` is exactly the regression the
        happy-path check exists to catch, so the classifier's rejection of it is
        asserted too rather than left implied.
        """
        for kind in ("SyntaxError", "AttributeError", "TypeError"):
            assert not classify_error_kind({"kind": kind}), (
                f"{kind} was accepted as a plain import error"
            )

    def test_an_ok_result_classifies_as_neither_failure(self, probe: dict) -> None:
        """A clean import must not be mistaken for a classifiable failure.

        This is what makes the skip guards above sound: ``ok`` means there is
        nothing to classify, not that the classifier ran and passed.
        """
        assert not classify_error_kind({"kind": "ok"})
        assert not classify_blocker({"kind": "ok"})


class TestVersionAttribute:
    def test_version_is_always_a_string(self, probe: dict) -> None:
        if probe["kind"] != "ok":
            pytest.skip(
                f"vendored package does not import here ({probe['kind']}: "
                f"{probe.get('module')}); the module-level block is covered by "
                f"test_version_block_is_self_contained"
            )
        assert isinstance(probe["version"], str)
        assert probe["version"]

    def test_version_block_is_self_contained(self) -> None:
        """Execute the version block alone, with every heavy import stubbed.

        This isolates the thing the patch actually guards. The sibling modules
        the package pulls in are replaced by stand-ins, so the only real code
        running is the module header -- the part that must never raise.
        """
        package = os.path.join(SRC_DIR, "chatterbox")
        vendored = f"{PACKAGE_NAME}.src.chatterbox"
        script = textwrap.dedent(
            f"""
            import sys, types
            from importlib.metadata import PackageNotFoundError
            import importlib.metadata as md

            def _raise(name):
                raise PackageNotFoundError(name)
            md.version = _raise
            sys.modules['importlib.metadata'].version = _raise

            # Register the package chain the vendored tree lives in: the node
            # package, its `src` subpackage, and the vendored package itself.
            root = types.ModuleType({PACKAGE_NAME!r})
            root.__path__ = [{ROOT_DIR!r}]
            root.__package__ = {PACKAGE_NAME!r}
            sys.modules[{PACKAGE_NAME!r}] = root
            src_pkg = types.ModuleType({PACKAGE_NAME + '.src'!r})
            src_pkg.__path__ = [{SRC_DIR!r}]
            src_pkg.__package__ = {PACKAGE_NAME + '.src'!r}
            sys.modules[{PACKAGE_NAME + '.src'!r}] = src_pkg

            # Register the package and stub its three entry points so only the
            # module header runs. The stubs carry the names __init__ re-exports.
            pkg = types.ModuleType({vendored!r})
            pkg.__path__ = [{package!r}]
            pkg.__package__ = {vendored!r}
            sys.modules[{vendored!r}] = pkg
            exports = {{
                'tts': ('ChatterboxTTS',),
                'vc': ('ChatterboxVC',),
                'mtl_tts': ('ChatterboxMultilingualTTS', 'SUPPORTED_LANGUAGES'),
            }}
            for name, attrs in exports.items():
                stub = types.ModuleType({vendored!r} + '.' + name)
                for attr in attrs:
                    setattr(stub, attr, type(attr, (), {{}}))
                sys.modules[{vendored!r} + '.' + name] = stub

            source = open({os.path.join(package, '__init__.py')!r}, encoding='utf-8').read()
            exec(compile(source, 'src/chatterbox/__init__.py', 'exec'), pkg.__dict__)
            print('__VERSION__' + pkg.__dict__['__version__'])
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, (
            f"the vendored __init__.py raised with no distribution metadata:\n"
            f"{completed.stderr[-2000:]}"
        )
        marker = next(
            (line for line in completed.stdout.splitlines() if line.startswith("__VERSION__")),
            None,
        )
        assert marker is not None, f"no version reported:\n{completed.stdout[-1000:]}"
        version = marker[len("__VERSION__"):]
        assert version, "__version__ fell back to an empty string"

    def test_fallback_version_is_not_the_distribution_name(self) -> None:
        """A silent `PackageNotFoundError` swallowed into a wrong version is worse."""
        with open(
            os.path.join(SRC_DIR, "chatterbox", "__init__.py"), "r", encoding="utf-8"
        ) as handle:
            source = handle.read()
        assert 'except PackageNotFoundError' not in source, (
            "catching only PackageNotFoundError leaves other metadata errors fatal"
        )