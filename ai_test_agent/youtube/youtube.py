#!/usr/bin/python3
# Intsall selenium > 4.17.x
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
import argparse
import time
from datetime import datetime, timedelta
import requests
import re
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from browser_repair import BrowserRepairAgent, parse_faults
import random
import platform
import logging
import sys

LOG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "youtube_test.log"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_log_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_fmt)
logger.addHandler(_console_handler)

_file_handler = logging.FileHandler(
    LOG_FILE_PATH,
    mode="w",
)
_file_handler.setFormatter(_log_fmt)
logger.addHandler(_file_handler)

# The repair module has its own logger; route it to the same console and file so
# every recipe attempt and LLM invocation is visible to the operator.
_repair_logger = logging.getLogger("browser_repair")
_repair_logger.setLevel(logging.INFO)
_repair_logger.propagate = False
if not _repair_logger.handlers:
    _repair_logger.addHandler(_console_handler)
    _repair_logger.addHandler(_file_handler)

def record_debug_error(context):
    """Record technical error details in the main client test log."""
    if sys.exc_info()[0] is None:
        logger.error(context)
    else:
        logger.exception(context)


def xpath_literal(value):
    """Return a safe XPath string literal for user-supplied visible text."""
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    parts = value.split('"')
    return "concat(" + ", '\"', ".join(f'"{part}"' for part in parts) + ")"


def upload_test_log(host, device_name):
    """Upload this client's log to the active YouTube test results directory."""
    logger.info("Uploading client log.")
    try:
        _file_handler.flush()
        with open(LOG_FILE_PATH, "rb") as log_file:
            response = requests.post(
                f"http://{host}:5002/upload_youtube_log",
                data={"device_name": device_name, "device_type": "client"},
                files={"log_file": ("youtube_test.log", log_file, "text/plain")},
                timeout=30,
            )
        if response.status_code != 200:
            logger.error(
                "Failed to upload YouTube test log. Status code: %s",
                response.status_code,
            )
    except Exception:
        logger.error(
            "Unable to upload the client log. The log remains available locally."
        )
        record_debug_error("Client log upload failed")


class YouTube(object):

    def __init__(self, url, resolution, host, port, duration, device_name, driver,
                 ai_fallback=False, locator_faults=None):
        self.url = url
        self.resolution = resolution
        self.device_name = device_name
        self.host = host
        self.port = port
        self.driver = driver
        self.duration = duration
        self.dataset = []
        self.stop_signal = False
        self.current_stage = "initializing the test"
        self.stats_upload_failures = 0
        self.server_connection_failed = False
        self.stop_check_failed = False
        self.last_progress_log = None
        self.browser_repair = (
            BrowserRepairAgent(driver, faults=locator_faults) if ai_fallback else None
        )

    def wait_for_element(
        self,
        locator,
        element_name,
        timeout=20,
        clickable=False,
        required=True,
    ):
        """Wait for an element and report failures with useful context."""
        requested_locator = locator
        if self.browser_repair is not None:
            locator = self.browser_repair.locator_for_run(locator, element_name)
            if self.browser_repair.is_faulted(element_name):
                timeout = min(timeout, 2)
                logger.info("Injected locator failure for %s: %s", element_name, locator[1])
        condition = (
            EC.element_to_be_clickable(locator)
            if clickable
            else EC.presence_of_element_located(locator)
        )

        try:
            return WebDriverWait(self.driver, timeout).until(condition)
        except TimeoutException:
            if self.browser_repair is not None:
                logger.warning(
                    "%s was not found with the active locator; trying saved and AI repairs.",
                    element_name,
                )
                repaired = self.browser_repair.repair(
                    requested_locator, element_name, clickable=clickable
                )
                if repaired is not None:
                    return repaired
            if required:
                logger.error(
                    "%s was not found within %s seconds.",
                    element_name,
                    timeout,
                )
                raise

            logger.warning(
                "%s was not found within %s seconds. Continuing without it.",
                element_name,
                timeout,
            )
            return None
        except Exception:
            logger.exception("Unexpected error while locating %s.", element_name)
            raise

    def simulate_human_movements(self):
        try:
            actions = ActionChains(self.driver)
            for _ in range(random.randint(2, 5)):
                actions.move_by_offset(random.randint(-20, 20), random.randint(-20, 20))
                actions.perform()
                time.sleep(random.uniform(0.1, 0.3))
            logger.info("Human mouse movements simulated successfully.")
        except Exception:
            logger.warning(
                "Unable to simulate mouse movement. Playback will continue normally."
            )

    def get_video_id(self):
        return self.url.split("=")[1]

    def load_video(self):
        logger.info("Loading YouTube video.")
        self.driver.get(self.url)
        return True

    def check_stop_signal(self):
        """Check the stop signal from the Flask server."""
        try:
            endpoint_url = f"http://{self.host}:5002/check_stop"

            response = requests.get(endpoint_url, timeout=5)
            if response.status_code == 200:
                if self.stop_check_failed:
                    logger.info("Connection to the test server recovered.")
                    self.stop_check_failed = False

                stop_signal_from_server = response.json().get("stop", False)

                # Only update if the server's stop signal is True
                if stop_signal_from_server:
                    self.stop_signal = True
                    logger.info("Stop signal received from server. Stopping playback.")
            return self.stop_signal
        except Exception:
            if not self.stop_check_failed:
                logger.warning(
                    "Unable to check for a stop request because the test server is unavailable."
                )
                self.stop_check_failed = True
                record_debug_error("Stop-signal request failed")
            return self.stop_signal

    def select_resolution(self):
        # Don't bother selecting resolution for Auto
        if self.resolution == "Auto":
            logger.info("Resolution set to Auto.")
            return True

        logger.info(f"Selecting resolution: {self.resolution}")
        time.sleep(0.2)
        sb = self.wait_for_element(
            (By.CSS_SELECTOR, ".ytp-button.ytp-settings-button"),
            "YouTube settings button",
            clickable=True,
        )
        self.driver.execute_script("arguments[0].click();", sb)
        time.sleep(0.3)
        quality = self.wait_for_element(
            (By.XPATH, "//*[contains(@class, 'ytp-menuitem-label') and normalize-space()='Quality']"),
            "YouTube Quality menu item",
            timeout=5,
            clickable=True,
        )
        self.driver.execute_script("arguments[0].click();", quality)
        time.sleep(0.3)
        try:
            resolution = self.wait_for_element(
                (By.XPATH, "//*[contains(@class, 'ytp-menuitem-label') and contains(normalize-space(), "
                 f"{xpath_literal(self.resolution)})]"),
                f"YouTube {self.resolution} resolution option",
                timeout=5,
                clickable=True,
                required=False,
            )
            if resolution is not None:
                self.driver.execute_script("arguments[0].click();", resolution)
                logger.info(f"Resolution set to {self.resolution}.")
                return True
            logger.warning(
                f"Resolution '{self.resolution}' not available. Falling back to Auto."
            )
            automatic = self.wait_for_element(
                (By.XPATH, "//*[contains(@class, 'ytp-menuitem-label') and contains(normalize-space(), 'Auto')]"),
                "YouTube Auto resolution option",
                timeout=5,
                clickable=True,
            )
            self.driver.execute_script("arguments[0].click();", automatic)
            return True
        except Exception:
            logger.error(
                "Unable to select resolution '%s'. The YouTube quality menu may not be available.",
                self.resolution,
            )
            record_debug_error("Resolution selection failed")
            return False

    def enable_stats(self):
        movie_player = self.wait_for_element(
            (By.CSS_SELECTOR, ".html5-video-container"),
            "YouTube video player",
            timeout=60,
        )
        movie_player = self.wait_for_element(
            (By.CSS_SELECTOR, ".html5-video-container"),
            "clickable YouTube video player",
            timeout=60,
            clickable=True,
        )
        self.hover = ActionChains(self.driver).move_to_element(movie_player)
        self.hover.perform()
        movie_player = self.wait_for_element(
            (By.CSS_SELECTOR, ".html5-video-container"),
            "clickable YouTube video player",
            timeout=60,
            clickable=True,
        )
        ActionChains(self.driver).context_click(movie_player).perform()
        option_child = self.wait_for_element(
            (By.XPATH, "//*[contains(@class, 'ytp-menuitem-label') and normalize-space()='Stats for nerds']"),
            "YouTube Stats for nerds menu item",
            timeout=5,
            clickable=True,
            required=False,
        )
        if option_child is not None:
            self.driver.execute_script("arguments[0].click();", option_child)
            logger.info("Stats for nerds panel enabled.")
            return True
        return False

    def get_stats(self):
        elem = self.wait_for_element(
            (By.CSS_SELECTOR, ".html5-video-info-panel-content.ytp-sfn-content"),
            "Stats for Nerds panel",
        )
        stats_data = elem.text
        stats_data = stats_data.replace(" ", "")
        viewport_match = re.search(
            r"Viewport/Frames([\d+x]+)(?:\*[\d.]+)?/([\d]+)droppedof([\d]+)", stats_data
        )
        current_optimal_res_match = re.search(
            r"Current/OptimalRes([\d@x]+)/([\d@x]+)", stats_data
        )
        # Initialize an empty dictionary to store extracted values
        data = {}
        # Check and assign the extracted values if matches were found
        if viewport_match:
            data["Viewport"] = viewport_match.group(1)  # e.g., '1280x720'
            data["DroppedFrames"] = viewport_match.group(2)  # e.g., '0'
            data["TotalFrames"] = viewport_match.group(3)  # e.g., '3930'

        if current_optimal_res_match:
            data["CurrentRes"] = current_optimal_res_match.group(
                1
            )  # e.g., '1920x1080@60'
            data["OptimalRes"] = current_optimal_res_match.group(
                2
            )  # e.g., '1920x1080@60'

        buffer_health_match = re.search(r"BufferHealth([\d.]+)s", stats_data)
        if buffer_health_match:
            data["BufferHealth"] = buffer_health_match.group(1)

        current_time = datetime.now().strftime("%H:%M:%S")
        data["Timestamp"] = current_time

        return data

    def get_current_seek(self):
        elem = self.wait_for_element(
            (By.CSS_SELECTOR, ".ytp-time-current"),
            "current playback time",
        )
        return elem.text

    def get_video_duration(self):
        elem = self.wait_for_element(
            (By.CSS_SELECTOR, ".ytp-time-duration"),
            "video duration",
        )
        return elem.text

    def enable_loop(self):
        movie_player = self.wait_for_element(
            (By.CSS_SELECTOR, ".html5-video-container"),
            "YouTube video player",
            clickable=True,
        )
        self.hover = ActionChains(self.driver).move_to_element(movie_player)
        self.hover.perform()
        ActionChains(self.driver).context_click(movie_player).perform()
        option_child = self.wait_for_element(
            (By.XPATH, "//*[contains(@class, 'ytp-menuitem-label') and normalize-space()='Loop']"),
            "YouTube Loop menu item",
            timeout=5,
            clickable=True,
            required=False,
        )
        if option_child is not None:
            self.driver.execute_script("arguments[0].click();", option_child)
            logger.info("Video loop playback enabled.")
            return True
        return False

    def full_screen(self):
        try:
            # YouTube auto-hides player controls. Reveal them before both the
            # deterministic lookup and any DOM-based repair candidate scan.
            player = self.wait_for_element(
                (By.CSS_SELECTOR, ".html5-video-container"),
                "YouTube video player for full-screen controls",
                timeout=5,
                required=False,
            )
            if player is not None:
                ActionChains(self.driver).move_to_element(player).perform()
                time.sleep(0.3)
            elem = self.wait_for_element(
                (By.CSS_SELECTOR, ".ytp-fullscreen-button.ytp-button"),
                "YouTube full-screen button",
                timeout=5,
                clickable=True,
                required=False,
            )
            if elem is None:
                return
            self.driver.execute_script("arguments[0].click();", elem)
        except Exception:
            logger.warning("Failed to enter full-screen mode.")

    def _run_stage(self, name, target, action, verify, required=True):
        """Run, verify, repair, and only then commit learned locator recipes."""
        if self.browser_repair is not None:
            replayed = self.browser_repair.try_saved_stage_recipe(name, verify)
            if replayed:
                return True
        result = False
        try:
            result = action()
        except Exception:
            logger.exception("Deterministic browser stage %s raised an error.", name)
        passed = result is not False and self._wait_for_verify(verify)
        if self.browser_repair is not None:
            self.browser_repair.confirm_pending_locators(passed)
        if passed:
            logger.info("Verified browser stage: %s.", name)
            return True

        logger.warning("Browser stage %s did not reach target state: %s", name, target)
        if self.browser_repair is not None:
            repaired = self.browser_repair.repair_stage(name, target, verify)
            self.browser_repair.confirm_pending_locators(repaired)
            if repaired:
                return True
        if required:
            logger.error("Required browser stage failed: %s.", name)
        return not required

    @staticmethod
    def _safe_verify(verify):
        try:
            return bool(verify())
        except Exception:
            return False

    @classmethod
    def _wait_for_verify(cls, verify, timeout=4, interval=0.2):
        """Allow asynchronous browser UI changes to settle before declaring failure."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cls._safe_verify(verify):
                return True
            time.sleep(interval)
        return cls._safe_verify(verify)

    def _video_loaded(self):
        return bool(self.driver.find_elements(By.CSS_SELECTOR, "video.html5-main-video"))

    def _stats_enabled(self):
        return any(element.is_displayed() for element in self.driver.find_elements(
            By.CSS_SELECTOR, ".html5-video-info-panel-content.ytp-sfn-content"))

    def _loop_enabled(self):
        return bool(self.driver.execute_script(
            "const v=document.querySelector('video'); return !!(v && v.loop);"))

    def _resolution_available(self):
        # The old contract accepts Auto when a requested quality is unavailable.
        return bool(self.driver.execute_script(
            "const p=document.getElementById('movie_player'); "
            "return !!(p && p.getPlaybackQuality && p.getPlaybackQuality());"))

    def _playback_advancing(self):
        before = self.driver.execute_script(
            "const v=document.querySelector('video'); return v ? v.currentTime : -1;")
        time.sleep(0.8)
        state = self.driver.execute_script(
            "const v=document.querySelector('video'); "
            "return v ? {time:v.currentTime, paused:v.paused, ended:v.ended} : null;")
        return bool(state and not state["paused"] and not state["ended"] and state["time"] > before)

    def _fullscreen_enabled(self):
        return bool(self.driver.execute_script(
            "const p=document.getElementById('movie_player'); "
            "return !!document.fullscreenElement || !!(p && p.classList.contains('ytp-fullscreen'));"))

    def play(self):
        self.current_stage = "loading the YouTube video"
        if not self._run_stage(
                "load_video", "YouTube video player is present",
                self.load_video, self._video_loaded):
            self.stop()
            return False
        self.simulate_human_movements()

        self.current_stage = "enabling Stats for Nerds"
        if not self._run_stage(
                "enable_stats", "Stats for Nerds panel is visible",
                self.enable_stats, self._stats_enabled):
            logger.error(
                "Unable to enable Stats for Nerds. Playback statistics cannot be collected."
            )
            self.stop()
            return False
        if self.duration:
            self.current_stage = "enabling video loop"
            if not self._run_stage(
                    "enable_loop", "HTML video loop mode is enabled",
                    self.enable_loop, self._loop_enabled):
                logger.error("Unable to enable video looping. The client test cannot continue.")
                self.stop()
                return False

        self.current_stage = "selecting the video resolution"
        if not self._run_stage(
                "select_resolution", "A playback quality is selected (requested quality or Auto fallback)",
                self.select_resolution, self._resolution_available):
            self.stop()
            return False

        self.video_duration = self.get_video_duration()
        time.sleep(1)
        self.current_stage = "playing the YouTube video"
        logger.info(
            "Playback started: duration=%s minute(s), resolution=%s.",
            self.duration,
            self.resolution,
        )
        if not self._run_stage(
                "start_playback", "Video is playing and currentTime advances",
                self.start, self._playback_advancing):
            self.stop()
            return False
        if not self._run_stage(
                "enter_fullscreen", "YouTube player is in fullscreen mode",
                self.full_screen, self._fullscreen_enabled):
            self.stop()
            return False
        if self.duration:
            end_time = datetime.now() + timedelta(minutes=self.duration)
            while datetime.now() <= end_time:
                if self.check_stop_signal():
                    break
                stats = self.get_stats()
                self.dataset.append(stats)
                self.send_stats_to_api(stats, self.device_name)
                time.sleep(1)
            stats = self.get_stats()
            self.dataset.append(stats)
            self.send_stats_to_api(stats, self.device_name, stop=True)
        else:
            time_array = self.video_duration.split(":")
            time_array = list(map(int, time_array))
            if len(time_array) == 3:
                delta = timedelta(
                    hours=time_array[0], minutes=time_array[1], seconds=time_array[2]
                )
            else:
                delta = timedelta(minutes=time_array[0], seconds=time_array[1])
            end_time = datetime.now() + timedelta(seconds=delta.total_seconds())
            while datetime.now() < end_time:
                stats = self.get_stats()
                self.dataset.append(stats)
                self.send_stats_to_api(stats, self.device_name)
                time.sleep(1)
            stats = self.get_stats()
            self.dataset.append(stats, self.device_name)
            self.send_stats_to_api(stats, stop=True)

        logger.info("Playback finished.")
        self.stop()
        return True

    def start(self):
        elem = self.wait_for_element(
            (By.CSS_SELECTOR, ".ytp-play-button.ytp-button"),
            "YouTube play button",
            clickable=True,
        )
        self.driver.execute_script(
            'if(arguments[0].dataset.titlenotooltip=="Play")arguments[0].click();',
            elem,
        )
        if elem.get_attribute("data-title-no-tooltip") != "Pause":
            self.driver.execute_script("arguments[0].click();", elem)

    def stop(self):
        logger.info("Closing browser driver.")
        self.driver.close()
        logger.info("Browser driver closed successfully.")

    def send_stats_to_api(self, stats, device_name, stop=False):
        try:
            url = f"http://{self.host}:5002/youtube_stats"
            headers = {
                "Content-Type": "application/json",
            }
            data = {
                device_name: stats,  # Device name as the key and stats as the value
                "stop": stop,  # Stop remains as a separate key
            }

            response = requests.post(url, json=data, headers=headers, timeout=5)
            if response.status_code == 200:
                if self.server_connection_failed:
                    logger.info("Connection to the test server recovered.")
                    self.server_connection_failed = False
            else:
                self._record_stats_upload_failure(
                    f"test server returned HTTP {response.status_code}"
                )
        except Exception:
            self._record_stats_upload_failure("test server request failed")

    def _record_stats_upload_failure(self, debug_context):
        self.stats_upload_failures += 1
        if not self.server_connection_failed:
            logger.warning(
                "Unable to send statistics to the test server. Playback will continue."
            )
            self.server_connection_failed = True
            record_debug_error(debug_context)


def main():

    parser = argparse.ArgumentParser(
        prog="youtube.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
        Youtube streaming automation
         """,
        description="""\
NAME: youtube.py
PURPOSE: This script will open youtube over browser and play it for mentioned duration or single loop and get the reporting stats

Example:
    # for running a single loop of any url with Auto video resolution

        python3 youtube.py --url <video url>

    # for running a single loop of any url with required video quality

        python3 youtube.py --url <video url> --res <resoultion>

    # for running a video for certain amount of time

        python3 youtube.py --url <video url> --duration <duration in minuites>
    PS: If we mentioned the duration the loop option of video will be enabled automatically

    --res <resolution>
         144p
         240p
         360p
         480p
         720p
         1080p
    if resoultion is not there in options then video will play with Auto quality

    --duration <duration in minitues>
         If we mentioned the duration the loop option of video will be enabled automatically
         and if not then video will play only once

    --env <enviornment variable>=<enviornment value>
        If required to assign multiple envirnment variable just use --env key=value again for other values
        example python3 youtube.py --env python=python3 --env DISPLAY=:0
          """,
    )

    parser.add_argument(
        "--url", type=str, default="https://www.youtube.com/watch?v=4GnVDPD01as"
    )
    parser.add_argument("--res", type=str, default="Auto")
    parser.add_argument("--duration", default=0, type=int)
    parser.add_argument("--env", action="extend", nargs="+", default=[])
    parser.add_argument(
        "--ai-fallback",
        action="store_true",
        help="Use saved recipes and the configured local LLM after a Selenium locator fails.",
    )
    parser.add_argument(
        "--inject-locator-failure",
        action="append",
        default=[],
        metavar="ELEMENT_KEY=CSS",
        help=("Replace a semantic locator for a repeatable healing test. Example: "
              "youtube_settings_button=.intentionally-broken"),
    )

    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=str, default=8000)
    parser.add_argument("--device_name", type=str, required=True)

    args = parser.parse_args()

    try:
        locator_faults = parse_faults(args.inject_locator_failure)
    except ValueError as exc:
        parser.error(str(exc))
    if locator_faults and not args.ai_fallback:
        parser.error("--inject-locator-failure requires --ai-fallback")

    logger.info("Test started: device=%s.", args.device_name)
    logger.info("Operating system: %s.", platform.system())

    for argument in args.env:
        arg = argument.split("=")
        os.environ[arg[0]] = arg[1]
    service = Service()
    options = webdriver.ChromeOptions()
    options.set_capability(
        "goog:loggingPrefs", {"performance": "ALL", "browser": "ALL"}
    )
    # # Performance-related options
    # options.add_argument("--process-per-site")
    # options.add_argument("--renderer-process-limit=4")
    # options.add_argument("--disable-background-timer-throttling")
    # options.add_argument("--disable-backgrounding-occluded-windows")
    # options.add_argument("--enable-low-res-tiling")
    # options.add_argument("--force-gpu-mem-available-mb=4096")
    # options.add_argument("--memory-pressure-thresholds-mb=2048")
    # options.add_experimental_option("detach", True)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    if platform.system() == "Windows":
        adguard_path = os.path.join(
            "C:\\",
            "Program Files (x86)",
            "LANforge-Server",
            "local",
            "real_application_test",
            "youtube",
            "adguard",
        )
    elif platform.system() == "Linux":
        adguard_path = "/home/lanforge/local/real_application_test/youtube/adguard"
    elif platform.system() == "Darwin":
        adguard_path = "/Users/lanforge/local/real_application_test/youtube/adguard"
    else:
        raise Exception("Unsupported OS")

    options.add_argument(f"load-extension={adguard_path}")

    yt = None
    driver = None
    result = "FAILED"
    failed_stage = "starting Chrome"
    try:
        logger.info("Starting Chrome.")
        driver = webdriver.Chrome(
            service=service, options=options
        )  # Or choose the appropriate webdriver for your browser

        yt = YouTube(
            args.url,
            args.res,
            args.host,
            args.port,
            args.duration,
            args.device_name,
            driver,
            ai_fallback=args.ai_fallback,
            locator_faults=locator_faults,
        )
        if args.duration != 0:
            yt.duration = args.duration
        if yt.play():
            result = "STOPPED BY USER" if yt.stop_signal else "COMPLETED"
        else:
            failed_stage = yt.current_stage
    except Exception:
        if yt is not None:
            failed_stage = yt.current_stage
        logger.error("YouTube test failed while %s.", failed_stage)
        record_debug_error(f"YouTube test failed while {failed_stage}")
    finally:
        if result == "FAILED" and driver is not None:
            logger.info("Stopping YouTube playback and closing Chrome.")
            try:
                driver.quit()
                logger.info("Chrome closed successfully.")
            except Exception:
                logger.warning(
                    "Chrome could not be closed completely. Manual cleanup may be required."
                )
                record_debug_error("Chrome cleanup failed")
        logger.info("TEST RESULT: %s", result)
        if yt is not None:
            logger.info(
                "Test summary: samples=%s, statistics upload failures=%s.",
                len(yt.dataset),
                yt.stats_upload_failures,
            )
        upload_test_log(
            args.host,
            args.device_name,
        )

    return 0 if result != "FAILED" else 1


if __name__ == "__main__":
    sys.exit(main())
