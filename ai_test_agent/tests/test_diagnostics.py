"""Regressions for false ad/Wi-Fi diagnoses and credential-safe run logs."""
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from openai import OpenAI
import diagnostics
import nim_client
from tools import adb_tools, youtube_tools


class FakeDriver:
    def __init__(self, xml, player=True):
        self.xml, self.player = xml, player
    def __call__(self, **selector):
        return SimpleNamespace(exists=self.player if "resourceId" in selector else False,
                               info={"bounds": {"left": 0, "top": 0, "right": 1080, "bottom": 700}})
    def dump_hierarchy(self):
        return self.xml


class DiagnosticTests(unittest.TestCase):
    def test_wifi_modern_status_connected_and_disconnected(self):
        result = adb_tools._parse_wifi_status('Wifi is enabled\nWifi is connected to "network"', '', False)
        self.assertIs(result['connected'], True)
        self.assertIs(result['wifi_enabled'], True)
        result = adb_tools._parse_wifi_status('Wifi is enabled\nWifi is not connected', 'state: CONNECTED', True)
        self.assertIs(result['connected'], False)

    def test_wifi_supplicant_current_state(self):
        result = adb_tools._parse_wifi_status('', 'mWifiInfo SSID: network, IP: /192.168.1.12, Supplicant state: COMPLETED', True)
        self.assertIs(result['connected'], True)
        result = adb_tools._parse_wifi_status('', 'mWifiInfo IP: /0.0.0.0, Supplicant state: COMPLETED', True)
        self.assertIsNone(result['connected'])

    def test_wifi_historical_connected_does_not_claim_association(self):
        result = adb_tools._parse_wifi_status('', 'history: state: CONNECTED\nold Wi-Fi is connected', True)
        self.assertIsNone(result['connected'])
        result = adb_tools._parse_wifi_status('', 'mNetworkInfo: state: DISCONNECTED/DISCONNECTED', True)
        self.assertIs(result['connected'], False)

    def test_enable_wifi_does_not_claim_network_connection(self):
        with patch.object(adb_tools, '_run_adb', return_value=SimpleNamespace(returncode=0, stderr='')):
            result = adb_tools.toggle_wifi('fake', True)
        self.assertTrue(result['command_accepted'])
        self.assertNotIn('connected', result)
        self.assertNotIn('wifi_enabled', result)

    def test_sponsored_card_outside_player_is_not_video_ad(self):
        driver = FakeDriver('<hierarchy><node resource-id="com.google.android.youtube:id/ad_badge" bounds="[0,900][300,950]"/></hierarchy>')
        self.assertEqual(youtube_tools._ad_evidence(driver), [])
        driver.xml = '<hierarchy><node resource-id="com.google.android.youtube:id/ad_badge" bounds="[0,100][300,150]"/></hierarchy>'
        self.assertEqual(len(youtube_tools._ad_evidence(driver)), 1)
        driver.xml = '<hierarchy><node resource-id="com.google.android.youtube:id/ad_card" bounds="[0,100][300,150]"/></hierarchy>'
        self.assertEqual(youtube_tools._ad_evidence(driver), [])

    def test_timeout_distinguishes_player_missing(self):
        with patch.object(youtube_tools, '_get_driver', return_value=FakeDriver('<hierarchy/>', player=False)):
            result = youtube_tools.wait_for_ads('fake', timeout_seconds=0)
        self.assertFalse(result['success'])
        self.assertIn('Player never became ready', result['error'])
        self.assertNotIn('Ads still showing', result['error'])

    def test_timeout_keeps_actual_ad_evidence(self):
        xml = '<hierarchy><node resource-id="com.google.android.youtube:id/modern_skip_ad_button" bounds="[200,100][400,200]"/></hierarchy>'
        with patch.object(youtube_tools, '_get_driver', return_value=FakeDriver(xml)):
            result = youtube_tools.wait_for_ads('fake', timeout_seconds=0)
        self.assertIn('Ads still showing', result['error'])
        self.assertEqual(len(result['ad_evidence']), 1)

    def test_parallel_run_logs_are_separate_and_redacted(self):
        def run(device):
            with diagnostics.diagnostic_run(device, 'test') as path:
                diagnostics.record('test_event', api_key='private', message='Bearer abc-secret nvapi-private',
                                   result={'_image_jpeg_b64': 'imagebytes', 'message': 'actual-key-value'})
                return path
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostics, 'LOG_DIR', Path(directory)), \
             patch.object(diagnostics, 'NVIDIA_NIM_API_KEY', 'actual-key-value'):
            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(pool.map(run, ['device1', 'device2']))
            self.assertNotEqual(*paths)
            for path, device in zip(paths, ['device1', 'device2']):
                text = Path(path).read_text()
                for secret in ('abc-secret', 'nvapi-private', 'actual-key-value', 'imagebytes'):
                    self.assertNotIn(secret, text)
                events = [json.loads(line) for line in text.splitlines()]
                self.assertTrue(all(event['device_id'] == device for event in events))
                self.assertEqual([event['event'] for event in events], ['run_started', 'test_event', 'run_ended'])

    def test_nvidia_500_logs_provider_details_and_request_id_without_key(self):
        def handler(request):
            return httpx.Response(500, json={'error': {'code': 'backend_failure', 'message': 'Failed with nvapi-private'}},
                                  headers={'x-request-id': 'req-test'})
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostics, 'LOG_DIR', Path(directory)):
            with diagnostics.diagnostic_run('device1', 'test') as path:
                with OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key='nvapi-private', max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(handler),
                                event_hooks={'request': [nim_client._request_log], 'response': [nim_client._response_log]})) as client:
                    with self.assertRaises(nim_client.NIMError) as raised:
                        nim_client.chat_completion(client, [{'role': 'user', 'content': 'Hi'}], stream=False)
                self.assertIn('req-test', str(raised.exception))
            text = Path(path).read_text()
            self.assertIn('backend_failure', text)
            self.assertIn('req-test', text)
            self.assertNotIn('nvapi-private', text)
            self.assertNotIn('authorization', text)

    def test_repair_failure_report_retains_root_cause(self):
        import play_youtube
        args = SimpleNamespace(stats_for_nerds=True, fullscreen=True, keep_playing=True, ai_fallback=True,
                               simulate=True, inject_failure=False, sim_unfamiliar_ui=False,
                               sim_stats_setting_off=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostics, "LOG_DIR", Path(directory)), \
             patch.object(play_youtube, "fix_step", return_value={"fixed": False, "summary": "NIM HTTP 500",
                                                                   "needs_human": True}), \
             patch.object(play_youtube.report_generator, "save_report", return_value="report.json"), \
             patch("builtins.print"):
            with diagnostics.diagnostic_run("fake", "test"):
                report = play_youtube.run_phone("fake", "https://youtu.be/test", 5, args)
            self.assertTrue(report["failure_observed"])
            self.assertIn("stats", report["root_cause"])
            self.assertEqual(report["status"], "NEEDS_HUMAN")
            self.assertTrue(Path(report["diagnostic_log"]).exists())

    def test_logcat_nonzero_is_a_failure_and_redacts_credentials(self):
        import log_collector
        result = SimpleNamespace(returncode=1, stderr="device offline nvapi-private", stdout="")
        with patch.object(log_collector.subprocess, "run", return_value=result):
            captured = log_collector.capture_failure("fake", "test")
        self.assertFalse(captured["success"])
        self.assertNotIn("nvapi-private", captured["error"])

    def test_retries_logged_and_success_does_not_duplicate_tools(self):
        attempts = []
        def handler(request):
            attempts.append(request)
            if len(attempts) < 3:
                return httpx.Response(500, json={'error': {'message': 'temporary failure'}}, headers={'retry-after': '0.01'})
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'test',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Hello'}}]})
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostics, 'LOG_DIR', Path(directory)):
            with diagnostics.diagnostic_run('device1', 'test') as path:
                with OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key='test', max_retries=2,
                            http_client=httpx.Client(transport=httpx.MockTransport(handler),
                                event_hooks={'request': [nim_client._request_log], 'response': [nim_client._response_log]})) as client:
                    message, _ = nim_client.chat_completion(client, [], stream=False)
                self.assertEqual(message['content'], 'Hello')
            events = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual([e['retry_number'] for e in events if e['event'] == 'nim_http_attempt'], ['0', '1', '2'])
            self.assertEqual([e['status'] for e in events if e['event'] == 'nim_http_response'], [500, 500, 200])


if __name__ == '__main__':
    unittest.main()
