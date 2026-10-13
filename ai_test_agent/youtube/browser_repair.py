"""Constrained, recipe-backed Selenium locator repair for the desktop test."""

import json
import logging
import re
import sys
import time
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from nim_client import LLMError, chat_completion, create_client  # noqa: E402


logger = logging.getLogger(__name__)
RECIPE_PATH = Path(__file__).resolve().with_name("browser_recipes.json")
STAGE_RECIPE_PATH = Path(__file__).resolve().with_name("browser_stage_recipes.json")
MAX_CANDIDATES = 80
MAX_STAGE_REPAIR_TURNS = 3
# Tool-capable reasoning models may use part of the completion budget for
# internal reasoning before emitting the required tool call.  A 512-token
# budget was too small for stage repair with a full DOM candidate list.
REPAIR_MAX_TOKENS = 2048


def semantic_key(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class BrowserRepairAgent:
    """Repair only failed element lookup calls on the current WebDriver session."""

    def __init__(self, driver, faults=None, recipe_path=RECIPE_PATH,
                 stage_recipe_path=STAGE_RECIPE_PATH):
        self.driver = driver
        self.faults = faults or {}
        self.recipe_path = Path(recipe_path)
        self.stage_recipe_path = Path(stage_recipe_path)
        self.pending_locator_repairs = []

    def locator_for_run(self, locator, element_name):
        """Apply an explicitly requested test fault before the original lookup."""
        replacement = self.faults.get(semantic_key(element_name))
        return (By.CSS_SELECTOR, replacement) if replacement else locator

    def is_faulted(self, element_name):
        return semantic_key(element_name) in self.faults

    def repair(self, original_locator, element_name, clickable=False):
        key = semantic_key(element_name)
        recipe = self._recipe(key)
        if recipe:
            element = self._locate(recipe["by"], recipe["value"], clickable, timeout=4)
            if element is not None:
                self.pending_locator_repairs.append({"key": key, "source": "saved"})
                logger.info("Saved browser recipe located %s; awaiting stage verification.", element_name)
                return element
            self._record_result(key, False)

        candidates = self._rank_candidates(element_name, self._visible_elements())
        if not candidates:
            logger.error("AI repair found no visible DOM candidates for %s.", element_name)
            return None

        selected = self._deterministic_candidate(element_name, candidates)
        if selected is not None:
            logger.info("Deterministic semantic repair matched DOM candidate %s for %s.",
                        selected[0], element_name)
        else:
            logger.info("Calling local AI to repair %s with %s DOM candidates.",
                        element_name, len(candidates))
            selected = self._ask_model(element_name, original_locator, clickable, candidates)
        if selected is None:
            return None
        index, reason = selected
        if index < 0 or index >= len(candidates):
            logger.error("AI selected an invalid DOM candidate for %s.", element_name)
            return None

        candidate = candidates[index]
        selector = self._usable_selector(candidate)
        logger.info("Repair selected DOM candidate %s for %s: selector=%s, reason=%s",
                    index, element_name, selector, reason)
        self._prepare_target(element_name)
        # Candidate visibility/clickability was established in the DOM snapshot.
        # YouTube controls may fade during model inference; return the present
        # element and let the enclosing action plus stage verifier decide success.
        element = self._locate(By.CSS_SELECTOR, selector, False, timeout=5)
        if element is None:
            logger.error("AI-selected locator did not verify for %s.", element_name)
            return None

        self.pending_locator_repairs.append({
            "key": key, "source": "ai", "original": original_locator,
            "selector": selector, "reason": reason,
        })
        logger.info("Repaired locator for %s with selector %s; awaiting stage verification.",
                    element_name, selector)
        return element

    def confirm_pending_locators(self, passed):
        """Commit AI locators only after the enclosing browser stage verifies."""
        pending, self.pending_locator_repairs = self.pending_locator_repairs, []
        for item in pending:
            if item["source"] == "saved":
                self._record_result(item["key"], passed)
            elif passed:
                self._save_recipe(item["key"], item["original"], item["selector"], item["reason"])
                logger.info("Verified and saved browser locator recipe %s.", item["key"])
            else:
                logger.warning("Discarded unverified AI locator for %s.", item["key"])

    def repair_stage(self, stage, target, verify):
        """Replay a verified stage recipe, then run a bounded browser action loop."""
        actions = []
        history = []
        for turn in range(MAX_STAGE_REPAIR_TURNS):
            self._prepare_target(target)
            candidates = self._rank_candidates(target, self._visible_elements())
            if not candidates:
                logger.error("Stage repair found no DOM candidates for %s.", stage)
                return False
            logger.info("Calling local AI for stage %s (turn %s/%s, %s candidates).",
                        stage, turn + 1, MAX_STAGE_REPAIR_TURNS, len(candidates))
            decision = self._ask_stage_model(stage, target, candidates, history)
            if decision is None:
                return False
            if decision["action"] == "cannot_repair":
                logger.error("AI could not repair stage %s: %s", stage, decision.get("reason", ""))
                return False
            result, saved_action = self._execute_stage_decision(decision, candidates)
            history.append({"decision": decision, "result": result})
            if not result.get("success"):
                continue
            if saved_action:
                actions.append(saved_action)
            if self._wait_until_verified(verify):
                self._save_stage_recipe(stage, target, actions)
                logger.info("AI repaired and verified browser stage %s.", stage)
                return True
        logger.error("AI stage repair exhausted its action limit for %s.", stage)
        return False

    def try_saved_stage_recipe(self, stage, verify):
        """Return None when absent, otherwise whether the verified replay passed."""
        recipe = self._stage_recipe(stage)
        if recipe is None:
            return None
        logger.info("Trying saved browser stage recipe for %s.", stage)
        passed = (self._run_stage_actions(recipe.get("actions", [])) and
                  self._wait_until_verified(verify))
        self._record_stage_result(stage, passed)
        if passed:
            logger.info("Saved browser stage recipe repaired and verified %s.", stage)
        else:
            logger.warning("Saved browser stage recipe failed verification for %s.", stage)
        return passed

    @staticmethod
    def _verified(verify):
        try:
            return bool(verify())
        except Exception:
            return False

    @classmethod
    def _wait_until_verified(cls, verify, timeout=4, interval=0.2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cls._verified(verify):
                return True
            time.sleep(interval)
        return cls._verified(verify)

    @staticmethod
    def _preferred_selector(candidate):
        """Prefer stable YouTube player classes over escaped attribute values."""
        classes = [value for value in candidate.get("classes", [])
                   if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", value)]
        youtube_classes = [value for value in classes if value.startswith("ytp-")]
        if youtube_classes:
            tag = candidate.get("tag") or ""
            return tag + "".join(f".{value}" for value in youtube_classes[:3])
        return candidate.get("selector")

    def _usable_selector(self, candidate):
        """Use a stable class selector only when it still identifies one element."""
        preferred = self._preferred_selector(candidate)
        original = candidate.get("selector")
        for selector in (preferred, original):
            if not isinstance(selector, str) or not selector:
                continue
            try:
                if len(self.driver.find_elements(By.CSS_SELECTOR, selector)) == 1:
                    return selector
            except Exception:
                continue
        return original or preferred

    @staticmethod
    def _deterministic_candidate(element_name, candidates):
        """Resolve unambiguous text/accessibility matches without a slow LLM call."""
        ignored = {"youtube", "button", "clickable", "current", "the", "for",
                   "controls", "menu", "item", "option", "panel", "video"}
        target = re.sub(r"[^a-z0-9]+", " ", element_name.lower()).strip()
        tokens = {token for token in target.split() if len(token) > 2 and token not in ignored}
        matches = []
        for index, candidate in enumerate(candidates):
            evidence = " ".join(str(candidate.get(key, "")) for key in
                                ("text", "aria_label", "title"))
            evidence += " " + " ".join(candidate.get("classes", []))
            haystack = re.sub(r"[^a-z0-9]+", " ", evidence.lower()).strip()
            matched = {token for token in tokens if token in haystack}
            exact_label = any(
                value and len(value.strip()) >= 4 and
                re.sub(r"[^a-z0-9]+", " ", value.lower()).strip() in target
                for value in (candidate.get("text", ""), candidate.get("aria_label", ""),
                              candidate.get("title", ""))
            )
            if (tokens and matched == tokens) or exact_label:
                matches.append((index, len(matched), exact_label))
        if not matches:
            return None
        matches.sort(key=lambda item: (item[1], item[2]), reverse=True)
        best = matches[0]
        if len(matches) > 1 and matches[1][1:] == best[1:]:
            return None
        return best[0], "unambiguous local semantic match"

    def _prepare_target(self, element_name):
        """Restore transient YouTube controls after a potentially slow model call."""
        if not any(word in element_name.lower() for word in ("full-screen", "fullscreen", "play button")):
            return
        try:
            player = self.driver.find_element(By.CSS_SELECTOR, ".html5-video-container")
            ActionChains(self.driver).move_to_element(player).perform()
        except Exception:
            logger.warning("Could not reveal player controls before verifying %s.", element_name)

    @staticmethod
    def _rank_candidates(element_name, candidates):
        ignored = {"youtube", "button", "clickable", "current", "the", "for", "controls"}
        target = re.sub(r"[^a-z0-9]+", " ", element_name.lower())
        tokens = [token for token in target.split() if len(token) > 2 and token not in ignored]

        def score(candidate):
            values = " ".join([
                candidate.get("tag", ""), candidate.get("text", ""),
                candidate.get("aria_label", ""), candidate.get("title", ""),
                candidate.get("role", ""), " ".join(candidate.get("classes", [])),
                candidate.get("selector", ""),
            ])
            haystack = re.sub(r"[^a-z0-9]+", " ", values.lower())
            matches = sum(1 for token in tokens if token in haystack)
            return (matches, int(candidate.get("clickable", False)),
                    int(bool(candidate.get("aria_label"))), int(bool(candidate.get("title"))))

        ranked = sorted(candidates, key=score, reverse=True)
        relevant = [candidate for candidate in ranked if score(candidate)[0] > 0]
        selected = (relevant[:30] if relevant else ranked[:30])
        return [{**candidate, "index": index} for index, candidate in enumerate(selected)]

    def _locate(self, by, value, clickable, timeout):
        condition = (EC.element_to_be_clickable((by, value)) if clickable
                     else EC.presence_of_element_located((by, value)))
        try:
            return WebDriverWait(self.driver, timeout).until(condition)
        except Exception:
            return None

    def _visible_elements(self):
        script = r"""
const esc = (value) => CSS.escape(value);
function selectorFor(el) {
  if (el.id) return '#' + esc(el.id);
  const ytpClasses = [...el.classList].filter(c => c.startsWith('ytp-'));
  if (ytpClasses.length) {
    const selector = el.tagName.toLowerCase() + ytpClasses.slice(0, 3).map(c => '.' + esc(c)).join('');
    if (document.querySelectorAll(selector).length === 1) return selector;
  }
  for (const attr of ['aria-label', 'data-tooltip-target-id', 'name', 'title']) {
    const value = el.getAttribute(attr);
    if (value && document.querySelectorAll(el.tagName.toLowerCase() + '[' + attr + '="' +
        CSS.escape(value) + '"]').length === 1) {
      return el.tagName.toLowerCase() + '[' + attr + '="' + CSS.escape(value) + '"]';
    }
  }
  const classes = [...el.classList].filter(c => !/^(active|selected|style-scope)$/.test(c));
  if (classes.length) {
    const selector = el.tagName.toLowerCase() + classes.slice(0, 3).map(c => '.' + esc(c)).join('');
    if (document.querySelectorAll(selector).length === 1) return selector;
  }
  const parts = [];
  let node = el;
  while (node && node.nodeType === 1 && parts.length < 6) {
    let part = node.tagName.toLowerCase();
    if (node.id) { parts.unshift('#' + esc(node.id)); break; }
    const siblings = node.parentElement ? [...node.parentElement.children].filter(x => x.tagName === node.tagName) : [];
    if (siblings.length > 1) part += ':nth-of-type(' + (siblings.indexOf(node) + 1) + ')';
    parts.unshift(part);
    node = node.parentElement;
  }
  return parts.join(' > ');
}
return [...document.querySelectorAll('body *')]
  .filter(el => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none';
  })
  .map(el => ({
    tag: el.tagName.toLowerCase(),
    text: (el.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 120),
    aria_label: el.getAttribute('aria-label') || '',
    title: el.getAttribute('title') || '',
    role: el.getAttribute('role') || '',
    classes: [...el.classList].slice(0, 8),
    selector: selectorFor(el),
    clickable: !!(el.matches('button,a,input,[role="button"],[role="menuitem"]') ||
                  getComputedStyle(el).cursor === 'pointer')
  }))
  .filter(item => item.clickable || item.aria_label || item.title || item.role ||
                  item.classes.some(c => c.startsWith('ytp-')) ||
                  (item.text && item.text.length <= 240))
  .sort((a, b) => {
    const score = item => (item.clickable ? 100 : 0) + (item.aria_label ? 80 : 0) +
      (item.title ? 40 : 0) + (item.role ? 30 : 0) +
      (item.classes.some(c => c.startsWith('ytp-')) ? 60 : 0);
    return score(b) - score(a);
  })
  .slice(0, arguments[0])
  .map((item, index) => ({index, ...item}));
"""
        try:
            values = self.driver.execute_script(script, MAX_CANDIDATES)
            return values if isinstance(values, list) else []
        except Exception:
            logger.exception("Unable to collect the DOM for AI locator repair.")
            return []

    def _ask_model(self, element_name, original_locator, clickable, candidates):
        tools = [
            {"type": "function", "function": {
                "name": "select_element",
                "description": "Select the one DOM candidate that matches the requested element.",
                "parameters": {"type": "object", "properties": {
                    "index": {"type": "integer"},
                    "reason": {"type": "string"},
                }, "required": ["index", "reason"]},
            }},
            {"type": "function", "function": {
                "name": "cannot_repair",
                "description": "Report that none of the supplied candidates safely match.",
                "parameters": {"type": "object", "properties": {
                    "reason": {"type": "string"},
                }, "required": ["reason"]},
            }},
        ]
        prompt = {
            "target": element_name,
            "must_be_clickable": clickable,
            "failed_locator": {"by": original_locator[0], "value": original_locator[1]},
            "url": self.driver.current_url,
            "title": self.driver.title,
            "visible_dom_candidates": candidates,
        }
        system = (
            "You repair one failed Selenium locator. Select only a candidate clearly matching the "
            "semantic target. Prefer accessibility labels, roles, titles, exact text, and YouTube "
            "player classes. Do not guess when evidence is weak. Call exactly one supplied tool."
        )
        try:
            with create_client() as client:
                message, refused = chat_completion(
                    client,
                    [{"role": "system", "content": system},
                     {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
                    tools=tools,
                    tool_choice="required",
                    stream=False,
                    max_tokens=REPAIR_MAX_TOKENS,
                )
        except LLMError as exc:
            logger.error("Browser locator repair could not call the LLM: %s", exc)
            return None
        calls = message.get("tool_calls", [])
        if refused or not calls:
            logger.error("Browser locator repair returned no decision for %s.", element_name)
            return None
        call = calls[0]
        try:
            arguments = json.loads(call["function"]["arguments"])
        except (KeyError, TypeError, ValueError):
            logger.error("Browser locator repair returned malformed arguments.")
            return None
        if call["function"]["name"] == "cannot_repair":
            logger.error("AI could not repair %s: %s", element_name, arguments.get("reason", "no match"))
            return None
        if call["function"]["name"] != "select_element" or not isinstance(arguments.get("index"), int):
            return None
        return arguments["index"], str(arguments.get("reason", "AI-selected semantic match"))

    def _ask_stage_model(self, stage, target, candidates, history):
        index_property = {"type": "integer", "minimum": 0, "maximum": len(candidates) - 1}
        tools = [
            self._tool_schema("click_element", "Click one visible DOM candidate.",
                              {"index": index_property, "reason": {"type": "string"}},
                              ["index", "reason"]),
            self._tool_schema("context_click_element", "Right-click one visible DOM candidate.",
                              {"index": index_property, "reason": {"type": "string"}},
                              ["index", "reason"]),
            self._tool_schema("hover_element", "Move the pointer over one visible DOM candidate.",
                              {"index": index_property, "reason": {"type": "string"}},
                              ["index", "reason"]),
            self._tool_schema("press_key", "Press one safe browser key.", {
                "key": {"type": "string", "enum": ["escape", "enter", "space", "f"]},
                "reason": {"type": "string"}}, ["key", "reason"]),
            self._tool_schema("cannot_repair", "Stop when no supplied action is safe.",
                              {"reason": {"type": "string"}}, ["reason"]),
        ]
        prompt = {
            "failed_stage": stage,
            "target_state": target,
            "url": self.driver.current_url,
            "title": self.driver.title,
            "previous_actions": history[-4:],
            "visible_dom_candidates": candidates,
        }
        system = (
            "Repair exactly one failed YouTube browser stage. Take one safe action from the supplied "
            "tools. Use candidate indexes only from the current DOM snapshot. Stay on YouTube, never "
            "click ads or external links, and do not repeat an action that failed. The runner verifies "
            "the target state after every successful action and will provide a fresh DOM next turn."
        )
        try:
            with create_client() as client:
                message, refused = chat_completion(
                    client,
                    [{"role": "system", "content": system},
                     {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
                    tools=tools, tool_choice="required", stream=False,
                    max_tokens=REPAIR_MAX_TOKENS)
        except LLMError as exc:
            logger.error("Browser stage repair could not call the LLM: %s", exc)
            return None
        calls = message.get("tool_calls", [])
        if refused or not calls:
            logger.error("Browser stage repair returned no action for %s.", stage)
            return None
        call = calls[0]
        try:
            arguments = json.loads(call["function"]["arguments"])
        except (KeyError, TypeError, ValueError):
            logger.error("Browser stage repair returned malformed arguments.")
            return None
        return {"action": call["function"]["name"], **arguments}

    @staticmethod
    def _tool_schema(name, description, properties, required):
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }}

    def _execute_stage_decision(self, decision, candidates):
        action = decision.get("action")
        if action == "press_key":
            keys = {"escape": Keys.ESCAPE, "enter": Keys.ENTER,
                    "space": Keys.SPACE, "f": "f"}
            key = decision.get("key")
            if key not in keys:
                return {"success": False, "error": "invalid key"}, None
            try:
                ActionChains(self.driver).send_keys(keys[key]).perform()
                return {"success": True}, {"action": "press_key", "key": key}
            except Exception as exc:
                return {"success": False, "error": type(exc).__name__}, None

        index = decision.get("index")
        if not isinstance(index, int) or not 0 <= index < len(candidates):
            return {"success": False, "error": "invalid candidate index"}, None
        selector = self._usable_selector(candidates[index])
        element = self._locate(By.CSS_SELECTOR, selector, action == "click_element", timeout=5)
        if element is None:
            return {"success": False, "error": "candidate no longer available"}, None
        try:
            if action == "click_element":
                self.driver.execute_script("arguments[0].click();", element)
            elif action == "context_click_element":
                ActionChains(self.driver).context_click(element).perform()
            elif action == "hover_element":
                ActionChains(self.driver).move_to_element(element).perform()
            else:
                return {"success": False, "error": "unsupported action"}, None
        except Exception as exc:
            return {"success": False, "error": type(exc).__name__}, None
        logger.info("AI stage action %s used selector %s: %s",
                    action, selector, decision.get("reason", ""))
        return {"success": True}, {"action": action, "selector": selector}

    def _run_stage_actions(self, actions):
        for action in actions:
            name = action.get("action")
            if name == "press_key":
                result, _ = self._execute_stage_decision(action, [])
            else:
                selector = action.get("selector")
                if not isinstance(selector, str):
                    return False
                candidate = {"selector": selector, "classes": [], "tag": ""}
                result, _ = self._execute_stage_decision(
                    {"action": name, "index": 0, "reason": "saved recipe"}, [candidate])
            if not result.get("success"):
                return False
            time.sleep(0.3)
        return True

    def _load(self):
        try:
            with self.recipe_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _recipe(self, key):
        recipe = self._load().get(key)
        if not isinstance(recipe, dict) or recipe.get("consecutive_failures", 0) >= 2:
            return None
        if recipe.get("by") != By.CSS_SELECTOR or not isinstance(recipe.get("value"), str):
            return None
        return recipe

    def _write(self, data):
        temporary = self.recipe_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self.recipe_path)

    def _record_result(self, key, passed):
        data = self._load()
        recipe = data.get(key)
        if not isinstance(recipe, dict):
            return
        recipe["successes"] = recipe.get("successes", 0) + int(passed)
        recipe["failures"] = recipe.get("failures", 0) + int(not passed)
        recipe["consecutive_failures"] = 0 if passed else recipe.get("consecutive_failures", 0) + 1
        self._write(data)

    def _save_recipe(self, key, original_locator, selector, reason):
        data = self._load()
        data[key] = {
            "by": By.CSS_SELECTOR,
            "value": selector,
            "original": {"by": original_locator[0], "value": original_locator[1]},
            "url_host": "youtube.com",
            "notes": reason,
            "successes": 1,
            "failures": 0,
            "consecutive_failures": 0,
        }
        self._write(data)

    def _load_stage_recipes(self):
        try:
            with self.stage_recipe_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write_stage_recipes(self, data):
        temporary = self.stage_recipe_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self.stage_recipe_path)

    def _stage_recipe(self, stage):
        recipe = self._load_stage_recipes().get(stage)
        if not isinstance(recipe, dict) or recipe.get("consecutive_failures", 0) >= 2:
            return None
        return recipe

    def _save_stage_recipe(self, stage, target, actions):
        if not actions:
            return
        data = self._load_stage_recipes()
        data[stage] = {"target": target, "actions": actions, "successes": 1,
                       "failures": 0, "consecutive_failures": 0}
        self._write_stage_recipes(data)

    def _record_stage_result(self, stage, passed):
        data = self._load_stage_recipes()
        recipe = data.get(stage)
        if not isinstance(recipe, dict):
            return
        recipe["successes"] = recipe.get("successes", 0) + int(passed)
        recipe["failures"] = recipe.get("failures", 0) + int(not passed)
        recipe["consecutive_failures"] = 0 if passed else recipe.get("consecutive_failures", 0) + 1
        self._write_stage_recipes(data)


def parse_faults(values):
    faults = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("locator faults must use semantic_key=css_selector")
        key, selector = value.split("=", 1)
        if not key.strip() or not selector.strip():
            raise ValueError("locator faults require a nonempty key and selector")
        faults[semantic_key(key)] = selector.strip()
    return faults
