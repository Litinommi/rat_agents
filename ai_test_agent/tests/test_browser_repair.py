import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "youtube"))

import browser_repair


class FakeDriver:
    current_url = "https://www.youtube.com/watch?v=test"
    title = "YouTube"

    def find_elements(self, _by, _selector):
        return [object()]


class BrowserRepairTests(unittest.TestCase):
    def test_faults_are_semantic_and_opt_in(self):
        faults = browser_repair.parse_faults([
            "YouTube settings button=.broken-settings",
        ])
        agent = browser_repair.BrowserRepairAgent(FakeDriver(), faults=faults)
        original = ("css selector", ".ytp-settings-button")
        self.assertEqual(agent.locator_for_run(original, "YouTube settings button"),
                         ("css selector", ".broken-settings"))
        self.assertTrue(agent.is_faulted("YouTube settings button"))

    def test_saved_recipe_is_used_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.json"
            path.write_text(json.dumps({"youtube_settings_button": {
                "by": "css selector", "value": ".working", "consecutive_failures": 0,
            }}), encoding="utf-8")
            expected = object()
            agent = browser_repair.BrowserRepairAgent(FakeDriver(), recipe_path=path)
            with patch.object(agent, "_locate", return_value=expected) as locate, \
                 patch.object(agent, "_ask_model") as model:
                actual = agent.repair(("css selector", ".broken"), "YouTube settings button", True)
            self.assertIs(actual, expected)
            locate.assert_called_once_with("css selector", ".working", True, timeout=4)
            model.assert_not_called()

    def test_verified_ai_selection_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.json"
            expected = object()
            agent = browser_repair.BrowserRepairAgent(FakeDriver(), recipe_path=path)
            candidates = [{"index": 0, "selector": "button[aria-label=Settings]"}]
            with patch.object(agent, "_visible_elements", return_value=candidates), \
                 patch.object(agent, "_ask_model", return_value=(0, "ARIA label matches")), \
                 patch.object(agent, "_locate", return_value=expected):
                actual = agent.repair(("css selector", ".broken"), "YouTube settings button", True)
            self.assertIs(actual, expected)
            agent.confirm_pending_locators(True)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["youtube_settings_button"]["value"],
                             "button[aria-label=Settings]")

    def test_candidates_are_ranked_by_semantic_target(self):
        candidates = [
            {"index": 0, "selector": ".ytp-settings-button", "classes": ["ytp-settings-button"],
             "clickable": True},
            {"index": 1, "selector": ".ytp-fullscreen-button", "classes": ["ytp-fullscreen-button"],
             "aria_label": "Full screen", "clickable": True},
        ]
        ranked = browser_repair.BrowserRepairAgent._rank_candidates(
            "YouTube full-screen button", candidates)
        self.assertEqual(ranked[0]["selector"], ".ytp-fullscreen-button")
        self.assertEqual(ranked[0]["index"], 0)

    def test_resolution_ranking_keeps_settings_and_does_not_match_autoplay(self):
        candidates = [
            {"selector": ".autoplay", "aria_label": "Autoplay", "classes": [],
             "clickable": True},
            {"selector": ".settings", "aria_label": "Settings", "classes": [],
             "clickable": True},
            {"selector": ".quality", "text": "Quality", "classes": [],
             "clickable": True},
        ]
        ranked = browser_repair.BrowserRepairAgent._rank_candidates(
            "A playback quality is selected (requested quality or Auto fallback)", candidates)
        self.assertEqual(ranked[0]["selector"], ".quality")
        self.assertIn(".settings", [candidate["selector"] for candidate in ranked])
        self.assertEqual(len(ranked), 3)

    def test_stage_decision_is_remapped_to_fresh_candidate_index(self):
        agent = browser_repair.BrowserRepairAgent(FakeDriver())
        old = [
            {"selector": ".other", "text": "Other", "classes": []},
            {"selector": ".settings", "aria_label": "Settings", "classes": []},
        ]
        fresh = [old[1], old[0]]
        with patch.object(agent, "_prepare_target"), \
             patch.object(agent, "_visible_elements", return_value=fresh):
            decision, candidates = agent._refresh_stage_decision(
                {"action": "click_element", "index": 1, "reason": "open settings"},
                "quality selected", old)
        self.assertEqual(decision["index"], 0)
        self.assertEqual(candidates[0]["selector"], ".settings")

    def test_ranking_preserves_generic_interaction_surfaces(self):
        candidates = [
            {"selector": ".settings", "aria_label": "Settings", "classes": [],
             "clickable": True, "area": 100},
            {"selector": ".large-surface", "classes": [], "clickable": False,
             "interaction_surface": True, "area": 500000},
        ]
        ranked = browser_repair.BrowserRepairAgent._rank_candidates(
            "unknown target state", candidates)
        self.assertIn(".large-surface", [candidate["selector"] for candidate in ranked])

    def test_exact_semantic_match_skips_model(self):
        candidates = [
            {"text": "Stats for nerds", "selector": ".stats", "classes": []},
            {"text": "Copy debug info", "selector": ".copy", "classes": []},
        ]
        self.assertEqual(
            browser_repair.BrowserRepairAgent._deterministic_candidate(
                "YouTube Stats for nerds menu item", candidates),
            (0, "unambiguous local semantic match"),
        )

    def test_force_ai_bypasses_saved_and_deterministic_locator_repairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.json"
            path.write_text(json.dumps({"youtube_quality_menu_item": {
                "by": "css selector", "value": ".saved", "consecutive_failures": 0,
            }}), encoding="utf-8")
            candidate = {"text": "Quality", "selector": ".quality", "classes": []}
            expected = object()
            agent = browser_repair.BrowserRepairAgent(
                FakeDriver(), recipe_path=path, force_ai=True)
            with patch.object(agent, "_visible_elements", return_value=[candidate]), \
                 patch.object(agent, "_ask_model", return_value=(0, "model chose Quality")) as model, \
                 patch.object(agent, "_locate", return_value=expected) as locate:
                actual = agent.repair(
                    ("xpath", "//broken"), "YouTube Quality menu item", True)
            self.assertIs(actual, expected)
            model.assert_called_once()
            locate.assert_called_once_with("css selector", ".quality", False, timeout=5)

    def test_force_ai_bypasses_saved_stage_recipe(self):
        agent = browser_repair.BrowserRepairAgent(FakeDriver(), force_ai=True)
        with patch.object(agent, "_stage_recipe") as recipe:
            self.assertIsNone(agent.try_saved_stage_recipe("enter_fullscreen", lambda: True))
        recipe.assert_not_called()

    def test_generic_youtube_class_does_not_replace_unique_selector(self):
        class NonUniqueDriver(FakeDriver):
            def find_elements(self, _by, selector):
                return [object(), object()] if selector == "div.ytp-menuitem" else [object()]

        agent = browser_repair.BrowserRepairAgent(NonUniqueDriver())
        candidate = {"tag": "div", "classes": ["ytp-menuitem"],
                     "selector": "div#menu > div:nth-of-type(7)"}
        self.assertEqual(agent._usable_selector(candidate),
                         "div#menu > div:nth-of-type(7)")

    def test_youtube_classes_replace_overescaped_attribute_selector(self):
        candidate = {
            "tag": "button",
            "classes": ["ytp-fullscreen-button", "ytp-button"],
            "selector": 'button[aria-label="Full\\ screen\\ \\(f\\)"]',
        }
        self.assertEqual(
            browser_repair.BrowserRepairAgent._preferred_selector(candidate),
            "button.ytp-fullscreen-button.ytp-button",
        )

    def test_unverified_ai_locator_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.json"
            agent = browser_repair.BrowserRepairAgent(FakeDriver(), recipe_path=path)
            candidate = {"index": 0, "selector": ".candidate", "classes": [], "tag": "button"}
            with patch.object(agent, "_visible_elements", return_value=[candidate]), \
                 patch.object(agent, "_ask_model", return_value=(0, "looks right")), \
                 patch.object(agent, "_locate", return_value=object()):
                agent.repair(("css selector", ".broken"), "Target", True)
            agent.confirm_pending_locators(False)
            self.assertFalse(path.exists())

    def test_stage_recipe_is_saved_only_after_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage-recipes.json"
            agent = browser_repair.BrowserRepairAgent(
                FakeDriver(), stage_recipe_path=path)
            candidate = {"index": 0, "selector": ".play", "classes": [], "tag": "button"}
            verified = [False, True]
            with patch.object(agent, "_visible_elements", return_value=[candidate]), \
                 patch.object(agent, "_ask_stage_model", return_value={
                     "action": "click_element", "index": 0, "reason": "play"}), \
                 patch.object(agent, "_execute_stage_decision", return_value=(
                     {"success": True}, {"action": "click_element", "selector": ".play"})):
                result = agent.repair_stage("start_playback", "video advances", lambda: verified.pop(0))
            self.assertTrue(result)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["start_playback"]["actions"][0]["selector"], ".play")

    def test_stage_model_receives_observed_action_result_on_next_turn(self):
        agent = browser_repair.BrowserRepairAgent(FakeDriver())
        candidate = {"selector": ".control", "classes": [], "tag": "button",
                     "text": "Control", "clickable": True}
        histories = []

        def decide(_stage, _target, _candidates, history):
            histories.append(list(history))
            return {"action": "click_element", "index": 0, "reason": "inspect control"}

        with patch.object(agent, "_visible_elements", return_value=[candidate]), \
             patch.object(agent, "_ask_stage_model", side_effect=decide), \
             patch.object(agent, "_execute_stage_decision", return_value=(
                 {"success": True}, {"action": "click_element", "selector": ".control"})), \
             patch.object(agent, "_wait_until_verified", side_effect=[False, True]), \
             patch.object(agent, "_save_stage_recipe"):
            self.assertTrue(agent.repair_stage("generic_stage", "target state", lambda: False))
        self.assertEqual(histories[0], [])
        self.assertEqual(histories[1][0]["selected"]["selector"], ".control")
        self.assertFalse(histories[1][0]["result"]["screen_changed"])

    def test_saved_stage_recipe_replays_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage-recipes.json"
            path.write_text(json.dumps({"enter_fullscreen": {
                "target": "fullscreen", "actions": [{"action": "press_key", "key": "f"}],
                "successes": 1, "failures": 0, "consecutive_failures": 0,
            }}), encoding="utf-8")
            agent = browser_repair.BrowserRepairAgent(FakeDriver(), stage_recipe_path=path)
            with patch.object(agent, "_run_stage_actions", return_value=True), \
                 patch.object(agent, "_ask_stage_model") as model:
                result = agent.try_saved_stage_recipe("enter_fullscreen", lambda: True)
            self.assertTrue(result)
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
