from pathlib import Path

from dynaconf import Dynaconf

fp = Path(__file__).parent / "settings.toml"
secrets = Path(__file__).parent / ".secrets.toml"
cfg = Dynaconf(
    envvar_prefix="CITYPY",
    settings_files=[fp, secrets],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
