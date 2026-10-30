"""S3.1 -- the frozen compatibility surface, captured BEFORE the teardown.

ComfyUI's frontend renders widgets in the order a node declares them and a
saved graph stores one value per widget **by index**. Nothing in the protocol
records the widget's name. Reorder the required inputs of a node and every
existing user graph keeps loading -- it just silently binds the wrong values,
e.g. ``use_watermark`` onto ``seed``. That failure is invisible until someone
synthesises with the watermark on, weeks later.

This module is the contract. It was written while the V1 implementation still
existed and its values were read out of it by hand; ``nodes.py`` was deleted
immediately afterwards. Steps S6.1/S6.2 rebuild the nodes on the V3 schema and
assert their ``define_schema`` output against ``frozen_widget_snapshot.json``.

Two independent sources of truth are asserted against each other here:

* the expected tuples embedded in this file (written from the V1 source), and
* ``tests/frozen_widget_snapshot.json`` (the durable artefact).

The shipped example workflow is the third: it is the ground truth for what real
saved graphs actually contain, and it is never rewritten to match us.
"""

from __future__ import annotations

import json
import os

import pytest

from conftest import ROOT_DIR

SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_widget_snapshot.json")
WORKFLOW_PATH = os.path.join(ROOT_DIR, "example_workflows", "ChatterboxTTS-workflow.json")

# (name, type, default, min, max, step, control_after_generate)
TTS_REQUIRED: tuple[tuple, ...] = (
    ("model_pack_name", "COMBO", "chatterbox", None, None, None, False),
    ("text", "STRING", "Hello, this is a test of Chatterbox TTS in ComfyUI.", None, None, None, False),
    ("max_new_tokens", "INT", 1000, 16, 4000, 8, False),
    ("flow_cfg_scale", "FLOAT", 0.7, 0.0, 2.0, 0.05, False),
    ("exaggeration", "FLOAT", 0.5, 0.25, 2.0, 0.05, False),
    ("temperature", "FLOAT", 0.8, 0.05, 5.0, 0.05, False),
    ("cfg_weight", "FLOAT", 0.5, 0.2, 1.0, 0.05, False),
    ("repetition_penalty", "FLOAT", 1.2, 1.0, 2.0, 0.1, False),
    ("min_p", "FLOAT", 0.05, 0.0, 1.0, 0.01, False),
    ("top_p", "FLOAT", 1.0, 0.0, 1.0, 0.01, False),
    ("seed", "INT", 0, 0, 0xFFFFFFFFFFFFFFFF, None, True),
    ("use_watermark", "BOOLEAN", False, None, None, None, False),
)

VC_REQUIRED: tuple[tuple, ...] = (
    ("model_pack_name", "COMBO", "chatterbox", None, None, None, False),
    ("source_audio", "AUDIO", None, None, None, None, False),
    ("n_timesteps", "INT", 10, 2, 50, 1, False),
    ("temperature", "FLOAT", 1.0, 0.0, 2.0, 0.05, False),
    ("flow_cfg_scale", "FLOAT", 0.7, 0.0, 2.0, 0.05, False),
    ("use_watermark", "BOOLEAN", False, None, None, None, False),
)

# Every key an input entry must carry, so a rewrite that drops one is loud.
INPUT_KEYS = frozenset(
    {
        "index",
        "name",
        "type",
        "default",
        "options",
        "options_source",
        "min",
        "max",
        "step",
        "multiline",
        "control_after_generate",
        "tooltip",
    }
)

# node id -> (RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY, OUTPUT_NODE)
NODE_ATTRIBUTES = {
    "ChatterboxTTS": (["AUDIO"], ["audio"], "synthesize", "audio/generation", True),
    # `converted_audio` is part of the frozen surface: user graphs wire the
    # output of this node by its displayed name.
    "ChatterboxVC": (["AUDIO"], ["converted_audio"], "convert_voice", "audio/generation", True),
}


@pytest.fixture(scope="module")
def snapshot() -> dict:
    assert os.path.isfile(SNAPSHOT_PATH), (
        f"the frozen snapshot is missing at {SNAPSHOT_PATH}; it must be captured "
        f"before the V1 implementation is deleted, never reconstructed after"
    )
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def signature(entry: dict) -> tuple:
    """The comparable part of an input entry: everything a re-bind would hit."""
    return (
        entry["name"],
        entry["type"],
        entry["default"],
        entry["min"],
        entry["max"],
        entry["step"],
        entry["control_after_generate"],
    )


class TestLegacySnapshotCaptured:
    """``frozen_widget_snapshot.json`` is well formed and says what it must."""

    def test_snapshot_file_exists_and_parses(self, snapshot: dict) -> None:
        assert isinstance(snapshot, dict)
        assert set(snapshot) >= {"captured", "captured_from", "nodes", "constants"}

    def test_snapshot_is_dated_and_attributed(self, snapshot: dict) -> None:
        assert snapshot["captured"] == "2026-10-03"
        assert "nodes.py" in snapshot["captured_from"]
        assert "V1" in " ".join(snapshot["_about"])

    def test_exactly_two_frozen_nodes(self, snapshot: dict) -> None:
        assert set(snapshot["nodes"]) == {"ChatterboxTTS", "ChatterboxVC"}

    @pytest.mark.parametrize("node_id", ["ChatterboxTTS", "ChatterboxVC"])
    def test_node_entry_is_complete(self, snapshot: dict, node_id: str) -> None:
        node = snapshot["nodes"][node_id]
        assert set(node) == {
            "class_name",
            "display_name",
            "RETURN_TYPES",
            "RETURN_NAMES",
            "FUNCTION",
            "CATEGORY",
            "OUTPUT_NODE",
            "required",
            "optional",
        }
        assert node["class_name"] in {"ChatterboxTTSNode", "ChatterboxVCNode"}
        assert node["display_name"]

    @pytest.mark.parametrize("node_id", ["ChatterboxTTS", "ChatterboxVC"])
    def test_every_input_entry_has_the_full_key_set(
        self, snapshot: dict, node_id: str
    ) -> None:
        node = snapshot["nodes"][node_id]
        for section in ("required", "optional"):
            for position, entry in enumerate(node[section]):
                assert set(entry) == INPUT_KEYS, f"{node_id}.{section}[{position}]"
                assert entry["index"] == position, (
                    f"{node_id}.{section}[{position}].index must be positional"
                )

    @pytest.mark.parametrize(
        ("node_id", "expected"),
        [("ChatterboxTTS", TTS_REQUIRED), ("ChatterboxVC", VC_REQUIRED)],
    )
    def test_required_inputs_match_the_v1_implementation(
        self, snapshot: dict, node_id: str, expected: tuple
    ) -> None:
        required = snapshot["nodes"][node_id]["required"]
        assert len(required) == len(expected), (
            f"{node_id} declared {len(required)} required inputs, the V1 "
            f"implementation declared {len(expected)}"
        )
        assert tuple(signature(entry) for entry in required) == expected

    def test_tts_required_order_is_the_frozen_one(self, snapshot: dict) -> None:
        names = [entry["name"] for entry in snapshot["nodes"]["ChatterboxTTS"]["required"]]
        assert names == [
            "model_pack_name",
            "text",
            "max_new_tokens",
            "flow_cfg_scale",
            "exaggeration",
            "temperature",
            "cfg_weight",
            "repetition_penalty",
            "min_p",
            "top_p",
            "seed",
            "use_watermark",
        ]

    def test_vc_required_order_is_the_frozen_one(self, snapshot: dict) -> None:
        names = [entry["name"] for entry in snapshot["nodes"]["ChatterboxVC"]["required"]]
        assert names == [
            "model_pack_name",
            "source_audio",
            "n_timesteps",
            "temperature",
            "flow_cfg_scale",
            "use_watermark",
        ]

    def test_seed_is_the_only_control_after_generate_input(self, snapshot: dict) -> None:
        """Exactly one flag is what makes 13 widgets out of 12 inputs."""
        flagged = [
            entry["name"]
            for node_id in ("ChatterboxTTS", "ChatterboxVC")
            for entry in snapshot["nodes"][node_id]["required"]
            if entry["control_after_generate"]
        ]
        assert flagged == ["seed"]

    def test_seed_range_is_unsigned_64_bit(self, snapshot: dict) -> None:
        seed = snapshot["nodes"]["ChatterboxTTS"]["required"][10]
        assert seed["name"] == "seed"
        assert seed["max"] == 0xFFFFFFFFFFFFFFFF

    def test_text_is_a_multiline_string(self, snapshot: dict) -> None:
        text = snapshot["nodes"]["ChatterboxTTS"]["required"][1]
        assert text["name"] == "text"
        assert text["type"] == "STRING"
        assert text["multiline"] is True

    def test_model_pack_options_are_dynamic_not_frozen(self, snapshot: dict) -> None:
        """The Combo is built from the models directory at schema-build time."""
        for node_id in ("ChatterboxTTS", "ChatterboxVC"):
            entry = snapshot["nodes"][node_id]["required"][0]
            assert entry["name"] == "model_pack_name"
            assert entry["type"] == "COMBO"
            assert entry["options"] is None
            assert entry["options_source"] == "get_chatterbox_model_pack_names()"

    @pytest.mark.parametrize("node_id", ["ChatterboxTTS", "ChatterboxVC"])
    def test_node_attributes_are_frozen(self, snapshot: dict, node_id: str) -> None:
        node = snapshot["nodes"][node_id]
        return_types, return_names, function, category, output_node = NODE_ATTRIBUTES[node_id]
        assert node["RETURN_TYPES"] == return_types
        assert node["RETURN_NAMES"] == return_names
        assert node["FUNCTION"] == function
        assert node["CATEGORY"] == category
        assert node["OUTPUT_NODE"] is output_node

    def test_vc_output_name_is_converted_audio(self, snapshot: dict) -> None:
        assert snapshot["nodes"]["ChatterboxVC"]["RETURN_NAMES"] == ["converted_audio"]

    def test_optional_inputs_are_frozen(self, snapshot: dict) -> None:
        assert [e["name"] for e in snapshot["nodes"]["ChatterboxTTS"]["optional"]] == [
            "audio_prompt"
        ]
        assert [e["name"] for e in snapshot["nodes"]["ChatterboxVC"]["optional"]] == [
            "target_voice_audio"
        ]


class TestCarriedForwardConstants:
    """S3.2 -- the handler's constants outlive the file that defined them.

    Batches that recreate them (S4.x downloader, S4.7 model discovery) compare
    their ``modules/constants.py`` against this record.
    """

    EXPECTED = {
        "CHATTERBOX_MODEL_SUBDIR": os.path.join("tts", "chatterbox"),
        "CHATTERBOX_REPO_ID": "ResembleAI/chatterbox",
        "CHATTERBOX_FILES_TO_DOWNLOAD": [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ],
        "DEFAULT_MODEL_PACK_NAME": "chatterbox",
    }

    def test_all_four_constants_are_recorded(self, snapshot: dict) -> None:
        assert set(snapshot["constants"]) - {"_about"} == set(self.EXPECTED)

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_constant_value_matches(self, snapshot: dict, name: str) -> None:
        recorded = snapshot["constants"][name]
        if name == "CHATTERBOX_MODEL_SUBDIR":
            # Recorded platform-neutrally: `value` is the POSIX rendering and
            # `parts` is the literal the replacement module must join.
            assert os.path.join(*recorded["parts"]) == self.EXPECTED[name]
            assert recorded["value"] == "tts/chatterbox"
        else:
            assert recorded["value"] == self.EXPECTED[name]
            assert "".join(recorded["parts"]) == "".join(
                part for part in self.EXPECTED[name]
            )

    def test_default_pack_name_is_the_only_official_voice(self, snapshot: dict) -> None:
        default = snapshot["constants"]["DEFAULT_MODEL_PACK_NAME"]["value"]
        assert default == "chatterbox"
        assert default in {
            entry["default"]
            for node in snapshot["nodes"].values()
            for entry in node["required"]
            if entry["name"] == "model_pack_name"
        }


class TestShippedWorkflowBindings:
    """The example workflow is what real saved graphs look like.

    It is the ground truth for the binding contract and is never edited to
    match the code -- if the code and this file disagree, the code is wrong.
    """

    @pytest.fixture(scope="class")
    def workflow(self) -> dict:
        assert os.path.isfile(WORKFLOW_PATH), f"missing {WORKFLOW_PATH}"
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _node(self, workflow: dict, node_type: str) -> dict:
        matches = [node for node in workflow.get("nodes", []) if node.get("type") == node_type]
        assert len(matches) == 1, f"expected exactly one {node_type} node in the example"
        return matches[0]

    def test_tts_stores_thirteen_widget_values(self, workflow: dict) -> None:
        values = self._node(workflow, "ChatterboxTTS")["widgets_values"]
        assert len(values) == 13

    def test_seed_companion_widget_is_at_index_11(self, workflow: dict) -> None:
        values = self._node(workflow, "ChatterboxTTS")["widgets_values"]
        assert values[11] == "randomize", (
            "index 11 is the control_after_generate widget ComfyUI emits after "
            "`seed`; anything else means `seed` moved or lost the flag"
        )

    def test_use_watermark_lands_at_index_12(self, workflow: dict) -> None:
        values = self._node(workflow, "ChatterboxTTS")["widgets_values"]
        assert values[12] is False

    def test_tts_widget_values_bind_the_frozen_inputs(self, workflow: dict, snapshot: dict) -> None:
        """Walk the stored values back through the frozen declaration.

        Two values are deliberately not compared against a default: index 10 is
        the ``seed``, which the author randomised when the graph was saved, and
        index 11 is the ``control_after_generate`` companion widget, which the
        node never declared. Every other position must hold the frozen default,
        because that is what the author left untouched.
        """
        values = self._node(workflow, "ChatterboxTTS")["widgets_values"]
        required = snapshot["nodes"]["ChatterboxTTS"]["required"]
        bound = [value for index, value in enumerate(values) if index != 11]
        assert len(bound) == len(required)
        for position, (value, entry) in enumerate(zip(bound, required)):
            if entry["name"] == "seed":
                assert isinstance(value, int) and entry["min"] <= value <= entry["max"]
                continue
            assert value == entry["default"], (
                f"widget index {position} of the saved graph is `{entry['name']}` "
                f"in the snapshot but holds {value!r}, not its frozen default "
                f"{entry['default']!r} -- the saved graph was written against a "
                f"different input order"
            )

    def test_vc_stores_five_widget_values(self, workflow: dict) -> None:
        values = self._node(workflow, "ChatterboxVC")["widgets_values"]
        assert len(values) == 5

    def test_vc_widget_values_bind_the_frozen_defaults(self, workflow: dict, snapshot: dict) -> None:
        values = self._node(workflow, "ChatterboxVC")["widgets_values"]
        required = snapshot["nodes"]["ChatterboxVC"]["required"]
        # `source_audio` is a socket: no widget value is stored for it.
        widget_entries = [e for e in required if e["type"] != "AUDIO"]
        assert len(values) == len(widget_entries) == 5
        for value, entry in zip(values, widget_entries):
            assert value == entry["default"], (
                f"saved graph holds {value!r} where the snapshot declares "
                f"`{entry['name']}` = {entry['default']!r}"
            )