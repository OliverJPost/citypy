import traceback

from typer.testing import CliRunner

from citypy.cli import app
from citypy.util.gpkg import GeoPackage

from .fixtures import mock_raw_building_data

runner = CliRunner()


def test_download(tmp_path, monkeypatch):
    # pwd to tmp_path as context manager
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["download", "Delfgauw", "NL"])

    if result.exception is not None:
        print("Exception occurred:")
        print(
            "".join(
                traceback.format_exception(
                    None, result.exception, result.exception.__traceback__
                )
            )
        )

    assert result.exit_code == 0

    gpkg_file = tmp_path / "Delfgauw_NL_data.gpkg"
    assert gpkg_file.exists()
    gpkg = GeoPackage(gpkg_file)
    assert "buildings" in gpkg.list_vector_layers()
