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
