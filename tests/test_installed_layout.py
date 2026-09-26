"""What `pip install agentfox` gets, which is not what a checkout gets.

`REPO_ROOT` is `parents[2]` of `src/agentfox/config.py`. From a checkout that is
the repository. From `<venv>/lib/python3.12/site-packages/agentfox/config.py` it
is `<venv>/lib/python3.12`. The expression never changes and what it names does,
so every path built on it is correct in CI and wrong for every user who installed
the package — which is exactly why none of this was caught before 0.3.1 went out.

Three of those paths mattered:

  - the database and the evidence packages were written *inside the virtualenv*,
    where a rebuild or `pip install --upgrade` discards them silently;
  - the migration scripts are at the repository root and not in the wheel, so
    `agentfox db upgrade` — "how a deployed instance is upgraded", per its own
    help — died with a raw alembic traceback naming a path in the user's venv;
  - nothing created the directory the SQLite file goes in, so the first command
    of a fresh install ended in `unable to open database file`.

These tests simulate the installed layout rather than requiring a built wheel, so
they run in the normal suite. tests/test_vendored_wheel_paths.py covers the wheel
itself.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# where a state root is chosen
# ---------------------------------------------------------------------------


def test_a_source_checkout_still_keeps_its_state_in_the_repository():
    """Contributors and scripts expect ./agentfox.db and ./var/evidence."""
    from agentfox.config import REPO_ROOT, state_root

    assert state_root() == REPO_ROOT
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_an_installed_package_does_not_write_inside_the_virtualenv(monkeypatch, tmp_path):
    """The failure this is here for: `<venv>/lib/python3.12/agentfox.db`.

    A governance database and signed auditor evidence in a directory that a venv
    rebuild throws away, with nothing said about it.
    """
    import agentfox.config as config

    venv_lib = tmp_path / "v" / "lib" / "python3.12"
    venv_lib.mkdir(parents=True)
    monkeypatch.setattr(config, "REPO_ROOT", venv_lib)
    monkeypatch.delenv(config.STATE_ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))

    root = config.state_root()
    assert venv_lib not in root.parents and root != venv_lib
    assert root.name == ".agentfox"


def test_xdg_data_home_is_honoured_when_it_is_set(monkeypatch, tmp_path):
    import agentfox.config as config

    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "nowhere")
    monkeypatch.delenv(config.STATE_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert config.state_root() == (tmp_path / "xdg" / "agentfox").resolve()


def test_the_state_directory_can_be_named_outright(monkeypatch, tmp_path):
    """An operator who puts this on a mounted volume must be able to say so."""
    import agentfox.config as config

    monkeypatch.setenv(config.STATE_ENV_VAR, str(tmp_path / "vol"))
    assert config.state_root() == (tmp_path / "vol").resolve()


def test_the_state_root_is_not_the_working_directory(monkeypatch, tmp_path):
    """`agentfox findings` must show the same findings from any directory."""
    import agentfox.config as config

    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "v" / "lib" / "python3.12")
    monkeypatch.delenv(config.STATE_ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    assert config.state_root() != Path(os.getcwd())


# ---------------------------------------------------------------------------
# the directory the database goes in
# ---------------------------------------------------------------------------


def test_a_sqlite_database_in_a_missing_directory_is_created_not_crashed(tmp_path):
    """`OperationalError: unable to open database file` was the whole first run."""
    from sqlalchemy import text

    from agentfox.db import _ensure_sqlite_directory

    target = tmp_path / "does" / "not" / "exist" / "agentfox.db"
    assert not target.parent.exists()
    _ensure_sqlite_directory(f"sqlite:///{target}")
    assert target.parent.is_dir()

    from sqlalchemy import create_engine

    with create_engine(f"sqlite:///{target}").connect() as conn:
        conn.execute(text("select 1"))
    assert target.is_file()


@pytest.mark.parametrize("url", ["sqlite://", "sqlite:///:memory:", "postgresql://h/d"])
def test_urls_with_no_file_behind_them_are_left_alone(url):
    from agentfox.db import _ensure_sqlite_directory

    _ensure_sqlite_directory(url)  # must not raise


# ---------------------------------------------------------------------------
# the migrations the wheel has to carry
# ---------------------------------------------------------------------------


def test_the_migrations_resolve_in_a_source_checkout():
    from agentfox.db import migration_root

    found = migration_root()
    assert found is not None
    ini, scripts = found
    assert ini.name == "alembic.ini" and (scripts / "env.py").is_file()


def test_the_migrations_resolve_from_the_package_when_the_repository_is_not_there(
    monkeypatch, tmp_path
):
    """The installed case: only the copy inside `agentfox/` exists."""
    import agentfox.config as config
    import agentfox.db as db

    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "not-a-checkout")

    packaged = Path(db.__file__).resolve().parent
    ini, scripts = packaged / "_alembic.ini", packaged / "_migrations"
    if not ini.is_file():  # a checkout has no copied-in pair; stand one in
        scripts.mkdir(exist_ok=True)
        ini.write_text("[alembic]\nscript_location = _migrations\n")
        created = True
    else:
        created = False
    try:
        found = db.migration_root()
        assert found == (ini, scripts)
    finally:
        if created:
            ini.unlink()
            scripts.rmdir()


def test_db_upgrade_says_what_is_wrong_instead_of_raising_an_alembic_traceback(
    monkeypatch, tmp_path
):
    """If the scripts are missing anyway, the message has to name the cause."""
    import agentfox.db as db

    monkeypatch.setattr(db, "migration_root", lambda: None)
    with pytest.raises(RuntimeError) as excinfo:
        db.upgrade_db()
    message = str(excinfo.value)
    assert "db upgrade" in message and "migration scripts" in message


# ---------------------------------------------------------------------------
# and the wheel actually carries them
# ---------------------------------------------------------------------------


def test_pyproject_force_includes_the_migrations_in_the_wheel():
    """Without this the resolution above has nothing to resolve to.

    A unit test can stand a directory in; only the build configuration puts the
    real one in the artefact a user installs.
    """
    from agentfox.config import REPO_ROOT

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    included = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["migrations"] == "agentfox/_migrations"
    assert included["alembic.ini"] == "agentfox/_alembic.ini"


# ---------------------------------------------------------------------------
# and the Docker build context carries them too
# ---------------------------------------------------------------------------


def test_every_dockerfile_that_installs_the_package_copies_the_forced_includes():
    """A force-include is a hard requirement of the build, not a nice-to-have.

    Adding `migrations`/`alembic.ini` to the wheel made `pip install .` fail
    anywhere those paths are absent from the build context. `deploy/Dockerfile`
    copies `pyproject.toml`, `README.md` and `src` and nothing else, so the
    gateway image stopped building the moment this merged:

        FileNotFoundError: Forced include not found: /app/alembic.ini
        ERROR: failed to solve: process "... pip install .[postgres,otel,classifiers]"

    The local wheel builds all passed, because the repository root has every
    path by definition. Only a build from a narrower context sees it — and CI's
    docker-smoke targets `deps`, which is the stage that runs the install.

    This is the same finding the Dockerfile already cites at its `deps` stage
    ("COPY paths pointing at directories that don't exist in the repo"),
    arriving from the other side: the COPY list is now short, not wrong.
    """
    import re
    import tomllib

    from agentfox.config import REPO_ROOT

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    forced = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    installs_the_package = re.compile(r"pip install\s+[\"']?\.")
    checked = 0
    for dockerfile in sorted(REPO_ROOT.glob("deploy/Dockerfile*")):
        text = dockerfile.read_text()
        if not installs_the_package.search(text):
            continue
        checked += 1
        copied = " ".join(
            line for line in text.splitlines() if line.startswith("COPY")
        )
        for source in forced:
            assert re.search(rf"(?<![\w/.-]){re.escape(source)}(?![\w/.-])", copied), (
                f"{dockerfile.name} runs `pip install .` but never COPYs "
                f"{source!r}, which pyproject force-includes into the wheel. "
                "The build fails with 'Forced include not found'."
            )

    assert checked, "no Dockerfile installs the package — did the deploy move?"
