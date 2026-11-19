"""The .env loader must fill missing variables but never shadow the shell."""
import os

from env import load_env_file


def test_loads_values_and_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("DFI_TEST_KEY", raising=False)
    monkeypatch.delenv("DFI_TEST_OTHER", raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n\nDFI_TEST_KEY=\"abc123\"\nexport DFI_TEST_OTHER=plain\n",
        encoding="utf-8",
    )
    assert load_env_file(path) is True
    assert os.environ["DFI_TEST_KEY"] == "abc123"
    assert os.environ["DFI_TEST_OTHER"] == "plain"


def test_shell_value_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("DFI_TEST_KEY", "from-shell")
    path = tmp_path / ".env"
    path.write_text("DFI_TEST_KEY=from-file\n", encoding="utf-8")
    load_env_file(path)
    assert os.environ["DFI_TEST_KEY"] == "from-shell"


def test_missing_file_is_silent(tmp_path):
    assert load_env_file(tmp_path / "absent.env") is False
