import json
from pathlib import Path

import pandas as pd

from src.viz.choropleth import load_boston_zip_boundaries, plot_zip_level_choropleth


def test_load_boston_zip_boundaries_uses_cached_geojson(tmp_path: Path):
    cache_path = tmp_path / "boston_zip_codes.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [],
    }
    cache_path.write_text(json.dumps(payload))

    loaded = load_boston_zip_boundaries(cache_path=cache_path)

    assert loaded == payload


def test_plot_zip_level_choropleth_writes_png_from_cached_geojson(tmp_path: Path):
    cache_path = tmp_path / "boston_zip_codes.geojson"
    cache_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"ZIP5": "02134"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"ZIP5": "02125"},
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": [
                                [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]]
                            ],
                        },
                    },
                ],
            }
        )
    )
    output_path = tmp_path / "student_housing_zip_context.png"
    summary = pd.DataFrame(
        {
            "zip_code": ["2134.0", "02125"],
            "total_violations": [10, 20],
        }
    )

    path = plot_zip_level_choropleth(
        summary,
        zip_col="zip_code",
        value_col="total_violations",
        output_path=output_path,
        title="Test ZIP Choropleth",
        legend_label="Violation Count",
        label_zip_codes=["02134"],
        boundary_cache_path=cache_path,
    )

    assert path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
