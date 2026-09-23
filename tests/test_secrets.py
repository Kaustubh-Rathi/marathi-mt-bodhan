"""Unit checks for the HF-token failover chain and the Kaggle pre-flight.

No network: every probe (``urlopen`` / ``hf_token_can_access``) is mocked, and
the developer's real tokens are cleared from the environment for each test.
Run from the repo root:  ``python -m unittest discover -s tests -v``
"""

from __future__ import annotations

import io
import os
import sys
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "kaggle"))

from mr_mt import secrets  # noqa: E402

try:  # kaggle_env is stdlib-only, but never let an import hiccup kill the suite
    import kaggle_env  # noqa: E402
except Exception:  # pragma: no cover - defensive
    kaggle_env = None  # type: ignore[assignment]

ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
GATED = ("datasets", "coild-aikosh/Education_v2")


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://huggingface.co/x", code, "err", None, None)


class _Resp:
    """Minimal stand-in for an ``urlopen`` response (context manager + status)."""

    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SecretsFailoverTest(unittest.TestCase):
    """Resolution order, dedupe and the gated-repo failover walk."""

    def setUp(self) -> None:
        self._saved = {key: os.environ.pop(key, None) for key in ENV_KEYS}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_env_token_used_when_no_probes_required(self) -> None:
        os.environ["HF_TOKEN"] = "env-tok"
        with mock.patch.object(secrets, "hf_token_can_access") as probe:
            token, source = secrets.select_hf_token(())
        self.assertEqual((token, source), ("env-tok", "env:HF_TOKEN"))
        probe.assert_not_called()

    def test_failover_bogus_env_token_to_dotenv(self) -> None:
        """THE regression guard: a bad env token must fall through to .env."""
        os.environ["HF_TOKEN"] = "bogus-env"
        verdicts = {"bogus-env": False, "good-file": True}
        probed: list = []

        def _probe(token, repo_type, repo_id, timeout=20):
            probed.append(token)
            return verdicts.get(token)

        with (
            mock.patch.object(secrets, "_dotenv_value", return_value="good-file"),
            mock.patch.object(secrets, "hf_token_can_access", side_effect=_probe),
            redirect_stderr(io.StringIO()) as err,
        ):
            token, source = secrets.select_hf_token([GATED])

        self.assertEqual((token, source), ("good-file", "dotenv:.env"))
        self.assertEqual(
            probed, ["bogus-env", "good-file"], "must walk the whole chain"
        )
        self.assertIn("env:HF_TOKEN", err.getvalue(), "rejection must be reported")

    def test_all_candidates_rejected_falls_back_to_first(self) -> None:
        os.environ["HF_TOKEN"] = "bogus-env"
        with (
            mock.patch.object(secrets, "_dotenv_value", return_value="also-bogus"),
            mock.patch.object(secrets, "hf_token_can_access", return_value=False),
            redirect_stderr(io.StringIO()) as err,
        ):
            token, source = secrets.select_hf_token([GATED])
        self.assertEqual((token, source), ("bogus-env", "env:HF_TOKEN"))
        self.assertIn("no candidate passed", err.getvalue())

    def test_no_candidates_at_all(self) -> None:
        with mock.patch.object(secrets, "_dotenv_value", return_value=None):
            token, source = secrets.select_hf_token([GATED])
        self.assertEqual((token, source), (None, "none"))

    def test_dedupe_yields_duplicate_value_once(self) -> None:
        os.environ["HF_TOKEN"] = "same"
        with mock.patch.object(secrets, "_dotenv_value", return_value="same"):
            pairs = list(secrets.iter_hf_tokens())
        self.assertEqual(pairs, [("env:HF_TOKEN", "same")])

    def test_get_env_secret_prefers_env_then_file(self) -> None:
        os.environ["MR_MT_TEST_KNOB"] = "env-key"
        try:
            self.assertEqual(secrets.get_env_secret("MR_MT_TEST_KNOB"), "env-key")
        finally:
            os.environ.pop("MR_MT_TEST_KNOB", None)
        with mock.patch.object(secrets, "_dotenv_value", return_value="file-key"):
            self.assertEqual(secrets.get_env_secret("MR_MT_TEST_KNOB"), "file-key")
        with mock.patch.object(secrets, "_dotenv_value", return_value=None):
            self.assertIsNone(secrets.get_env_secret("MR_MT_TEST_KNOB"))

    def test_get_hf_token_no_required_returns_first_without_probing(self) -> None:
        os.environ["HF_TOKEN"] = "env-tok"
        with (
            mock.patch.object(secrets, "_dotenv_value", return_value="file-tok"),
            mock.patch.object(
                secrets,
                "hf_token_can_access",
                side_effect=AssertionError("must not probe"),
            ),
            mock.patch.object(
                secrets,
                "select_hf_token",
                side_effect=AssertionError("must not delegate"),
            ),
        ):
            self.assertEqual(secrets.get_hf_token(), "env-tok")
            self.assertEqual(secrets.get_hf_token(()), "env-tok")
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        with mock.patch.object(secrets, "_dotenv_value", return_value=None):
            self.assertIsNone(secrets.get_hf_token())

    def test_get_hf_token_required_delegates_and_fails_over(self) -> None:
        os.environ["HF_TOKEN"] = "bogus-env"
        verdicts = {"bogus-env": False, "good-file": True}

        def _probe(token, repo_type, repo_id, timeout=20):
            return verdicts.get(token)

        with (
            mock.patch.object(secrets, "_dotenv_value", return_value="good-file"),
            mock.patch.object(secrets, "hf_token_can_access", side_effect=_probe),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(secrets.get_hf_token(required=[GATED]), "good-file")
            # also works positionally
            self.assertEqual(secrets.get_hf_token([GATED]), "good-file")

    def test_get_hf_token_required_no_candidate_returns_none(self) -> None:
        with (
            mock.patch.object(secrets, "_dotenv_value", return_value=None),
            redirect_stderr(io.StringIO()),
        ):
            self.assertIsNone(secrets.get_hf_token(required=[GATED]))

    def test_dotenv_distinct_candidate_preserved_when_values_differ(self) -> None:
        os.environ["HF_TOKEN"] = "env-tok"
        with mock.patch.object(secrets, "_dotenv_value", return_value="file-tok"):
            pairs = list(secrets.iter_hf_tokens())
        self.assertEqual(
            pairs, [("env:HF_TOKEN", "env-tok"), ("dotenv:.env", "file-tok")]
        )

    def test_dotenv_parser_export_spaces_quotes_comments(self) -> None:
        import tempfile

        content = (
            "# leading comment\n"
            "\n"
            "PLAIN=abc\n"
            "SPACED = spaced-val\n"
            "SPACED2=  spaced2  \n"
            "EXPORTED=ignored\n"
            "export EXPORTED=exported-val\n"
            "export  SPACED_EXPORT  =  exported-spaced\n"
            'DQUOTED="quoted-val" # trailing comment\n'
            "SQUOTED='single-val'  # another\n"
            "COMMENTED=real-val # strip me\n"
            "COMMENTED2=real2#kept\n"
            "EMPTY=\n"
            "NOEQUALS\n"
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".env", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(content)
            tmp = fh.name
        try:
            parsed = dict(secrets._iter_env_file(tmp))
            dq = secrets._dotenv_value("DQUOTED", tmp)
            cm = secrets._dotenv_value("COMMENTED", tmp)
            missing = secrets._dotenv_value("MISSING", tmp)
        finally:
            os.unlink(tmp)
        self.assertEqual(parsed.get("PLAIN"), "abc")
        self.assertEqual(parsed.get("SPACED"), "spaced-val")
        self.assertEqual(parsed.get("SPACED2"), "spaced2")
        self.assertEqual(parsed.get("EXPORTED"), "exported-val")
        self.assertEqual(parsed.get("SPACED_EXPORT"), "exported-spaced")
        self.assertEqual(parsed.get("DQUOTED"), "quoted-val")
        self.assertEqual(parsed.get("SQUOTED"), "single-val")
        self.assertEqual(parsed.get("COMMENTED"), "real-val")
        self.assertEqual(parsed.get("COMMENTED2"), "real2#kept")
        self.assertNotIn("NOEQUALS", parsed)
        self.assertEqual(dq, "quoted-val")
        self.assertEqual(cm, "real-val")
        self.assertIsNone(missing)

    def test_get_env_secret_parses_export_quotes_comments(self) -> None:
        body = 'export MR_MT_TEST_KNOB="file-val" # comment\n'
        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "read_text", return_value=body),
        ):
            self.assertEqual(secrets.get_env_secret("MR_MT_TEST_KNOB"), "file-val")
        body2 = "MR_MT_TEST_KNOB = bare-val  # strip\n"
        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "read_text", return_value=body2),
        ):
            self.assertEqual(secrets.get_env_secret("MR_MT_TEST_KNOB"), "bare-val")


class ProbeSemanticsTest(unittest.TestCase):
    """``hf_token_can_access`` must only fail on a definitive 401/403."""

    def test_readable_returns_true(self) -> None:
        with mock.patch("urllib.request.urlopen", return_value=_Resp(200)) as probe:
            self.assertIs(secrets.hf_token_can_access("t", "datasets", "x/y"), True)
        self.assertEqual(probe.call_count, 1)

    def test_401_and_403_return_false_only_after_all_files(self) -> None:
        # A missing file inside a gated repo also returns 403, so a single
        # 401/403 is not definitive: every candidate must be tried before
        # reporting blocked.
        for code in (401, 403):
            with self.subTest(code=code):
                with mock.patch(
                    "urllib.request.urlopen", side_effect=_http_error(code)
                ) as probe:
                    result = secrets.hf_token_can_access("t", "datasets", "x/y")
                self.assertIs(result, False)
                self.assertEqual(probe.call_count, len(secrets._PROBE_FILES))

    def test_403_then_readable_file_is_accessible(self) -> None:
        # The coild regression: config.json (absent) -> 403, README.md -> 200.
        with mock.patch(
            "urllib.request.urlopen", side_effect=[_http_error(403), _Resp(200)]
        ) as probe:
            self.assertIs(secrets.hf_token_can_access("t", "datasets", "x/y"), True)
        self.assertEqual(probe.call_count, 2)

    def test_404_falls_through_to_next_filename(self) -> None:
        with mock.patch(
            "urllib.request.urlopen", side_effect=[_http_error(404), _Resp(200)]
        ) as probe:
            self.assertIs(secrets.hf_token_can_access("t", "models", "x/y"), True)
        self.assertEqual(probe.call_count, 2)

    def test_all_404_is_undecidable_not_failure(self) -> None:
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[_http_error(404)] * len(secrets._PROBE_FILES),
        ) as probe:
            self.assertIsNone(secrets.hf_token_can_access("t", "models", "x/y"))
        self.assertEqual(probe.call_count, len(secrets._PROBE_FILES))

    def test_network_error_is_undecidable_not_failure(self) -> None:
        with mock.patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("offline")
        ) as probe:
            self.assertIsNone(secrets.hf_token_can_access("t", "models", "x/y"))
        self.assertEqual(
            probe.call_count,
            len(secrets._PROBE_FILES),
            "offline box must not reject every candidate",
        )


class KaggleEnvPreFlightTest(unittest.TestCase):
    """Stack gating, the access verdict and the ``--check-access`` CLI."""

    @classmethod
    def setUpClass(cls) -> None:
        if kaggle_env is None:  # pragma: no cover - defensive
            raise unittest.SkipTest("kaggle_env could not be imported")

    def test_required_gated_stacks(self) -> None:
        common = [
            ("datasets", "coild-aikosh/Education_v2"),
            ("datasets", "ai4bharat/IN22-Gen"),
            ("datasets", "facebook/flores"),
        ]
        self.assertEqual(
            kaggle_env._required_gated("bodhan"),
            common
            + [
                ("models", "bodhan-ai/indic-translate"),
                ("models", "google/gemma-4-E4B-it"),
            ],
        )
        self.assertEqual(
            kaggle_env._required_gated("indictrans2"),
            common + [("models", "ai4bharat/indictrans2-indic-indic-dist-320M")],
        )
        self.assertEqual(kaggle_env._required_gated("eval"), common)

    def test_verify_hf_access_with_provided_token(self) -> None:
        verdicts = {
            ("datasets", "coild-aikosh/Education_v2"): False,
            ("datasets", "ai4bharat/IN22-Gen"): True,
            ("datasets", "facebook/flores"): True,
            ("models", "bodhan-ai/indic-translate"): True,
            ("models", "google/gemma-4-E4B-it"): True,
        }
        with mock.patch.object(
            secrets,
            "hf_token_can_access",
            side_effect=lambda t, rt, rid, timeout=20: verdicts[(rt, rid)],
        ):
            out = kaggle_env.verify_hf_access("bodhan", token="tok")
        self.assertEqual(out["source"], "provided")
        self.assertIs(out["results"]["datasets/coild-aikosh/Education_v2"], False)
        self.assertIs(out["results"]["models/bodhan-ai/indic-translate"], True)

    def test_verify_hf_access_resolves_token_when_not_given(self) -> None:
        with (
            mock.patch.object(
                secrets, "select_hf_token", return_value=("resolved", "env:HF_TOKEN")
            ) as select,
            mock.patch.object(secrets, "hf_token_can_access", return_value=True),
        ):
            out = kaggle_env.verify_hf_access("bodhan")
        select.assert_called_once()
        self.assertEqual((out["token"], out["source"]), ("resolved", "env:HF_TOKEN"))

    def test_report_access_lists_only_blocked_repos(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf):
            kaggle_env._report_access(
                {
                    "datasets/coild-aikosh/Education_v2": False,
                    "models/bodhan-ai/indic-translate": True,
                }
            )
        text = buf.getvalue()
        self.assertIn("ACTION REQUIRED", text)
        self.assertIn("datasets/coild-aikosh/Education_v2", text)
        self.assertNotIn("models/bodhan-ai", text)
        buf = io.StringIO()
        with redirect_stderr(buf):
            kaggle_env._report_access({"datasets/coild-aikosh/Education_v2": True})
        self.assertEqual(buf.getvalue(), "", "all-OK must stay silent")

    def test_activate_fails_fast_on_blocked_repo(self) -> None:
        """A definite 401/403 must abort BEFORE the pip install, not at download."""
        verdict = {
            "token": "tok",
            "source": "env:HF_TOKEN",
            "results": {
                "datasets/coild-aikosh/Education_v2": False,
                "models/bodhan-ai/indic-translate": None,
            },
        }
        with (
            mock.patch.object(kaggle_env, "_install_deps") as install,
            mock.patch.object(kaggle_env, "_configure_rclone"),
            mock.patch.object(
                secrets, "select_hf_token", return_value=("tok", "env:HF_TOKEN")
            ),
            mock.patch.object(kaggle_env, "verify_hf_access", return_value=verdict),
            mock.patch.dict(os.environ, {"MR_MT_TOKEN_PROBE": "1"}),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as err,
        ):
            with self.assertRaises(SystemExit) as cm:
                kaggle_env.activate("bodhan")
        install.assert_not_called()
        self.assertIn("coild-aikosh/Education_v2", str(cm.exception))
        self.assertIn("ACTION REQUIRED", err.getvalue())

    def test_activate_fails_fast_without_any_token(self) -> None:
        """No candidate at all + probe on => abort (gated downloads would 401)."""
        with (
            mock.patch.object(kaggle_env, "_install_deps") as install,
            mock.patch.object(kaggle_env, "_configure_rclone"),
            mock.patch.object(secrets, "select_hf_token", return_value=(None, "none")),
            mock.patch.object(kaggle_env, "verify_hf_access") as verify,
            mock.patch.dict(os.environ, {"MR_MT_TOKEN_PROBE": "1"}),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as cm:
                kaggle_env.activate("bodhan")
        install.assert_not_called()
        verify.assert_not_called()
        self.assertIn("no HF token candidate", str(cm.exception))

    def test_activate_proceeds_when_verdict_unknown(self) -> None:
        """`None` (404/offline) is not a failure — the run must continue."""
        verdict = {
            "token": "tok",
            "source": "env:HF_TOKEN",
            "results": {"datasets/coild-aikosh/Education_v2": None},
        }
        with (
            mock.patch.object(kaggle_env, "_install_deps") as install,
            mock.patch.object(kaggle_env, "_configure_rclone") as rclone,
            mock.patch.object(
                secrets, "select_hf_token", return_value=("tok", "env:HF_TOKEN")
            ),
            mock.patch.object(kaggle_env, "verify_hf_access", return_value=verdict),
            mock.patch.dict(os.environ, {"MR_MT_TOKEN_PROBE": "1"}),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            repo = kaggle_env.activate("bodhan")
        install.assert_called_once_with("bodhan")
        rclone.assert_called_once()
        self.assertTrue((repo / "src" / "mr_mt").is_dir())

    def test_cli_exit_code_reflects_verdict(self) -> None:
        blocked = {
            "token": "t",
            "source": "dotenv:.env",
            "results": {"datasets/coild-aikosh/Education_v2": False},
        }
        ok = {
            "token": "t",
            "source": "dotenv:.env",
            "results": {"datasets/coild-aikosh/Education_v2": True},
        }
        with (
            mock.patch.object(kaggle_env, "verify_hf_access", return_value=blocked),
            redirect_stdout(io.StringIO()) as out,
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(kaggle_env._cli(["--check-access", "bodhan"]), 1)
        self.assertIn("BLOCKED", out.getvalue())
        with (
            mock.patch.object(kaggle_env, "verify_hf_access", return_value=ok),
            redirect_stdout(io.StringIO()) as out,
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(kaggle_env._cli(["--check-access", "bodhan"]), 0)
        self.assertIn("OK", out.getvalue())

    def test_ensure_import_path(self) -> None:
        repo = kaggle_env._ensure_import_path()
        self.assertTrue((repo / "src" / "mr_mt").is_dir())
        self.assertIn(str(repo / "src"), sys.path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
