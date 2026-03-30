"""Lightweight choropleth helpers backed by official Boston GIS GeoJSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from src.viz.plot_utils import plt, save_figure
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

from src.data.features import normalize_zip


BOSTON_ZIP_BOUNDARY_URL = (
    "https://gisportal.boston.gov/arcgis/rest/services/Planning/OpenData/MapServer/1/query"
    "?where=1%3D1&outFields=ZIP5&returnGeometry=true&f=geojson"
)
BOSTON_ZIP_BOUNDARY_CACHE_PATH = Path("data/raw/boston_zip_codes.geojson")


def _is_feature_collection(payload: dict[str, Any]) -> bool:
    return payload.get("type") == "FeatureCollection" and isinstance(payload.get("features"), list)


def load_boston_zip_boundaries(
    *,
    cache_path: Path = BOSTON_ZIP_BOUNDARY_CACHE_PATH,
    url: str = BOSTON_ZIP_BOUNDARY_URL,
    timeout: int = 60,
) -> dict[str, Any]:
    """Load official Boston ZIP boundaries from cache or the ArcGIS GeoJSON endpoint."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if _is_feature_collection(cached):
            return cached

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not _is_feature_collection(payload):
        raise ValueError("Boston ZIP boundary endpoint returned an unexpected payload.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload


def _outer_rings(geometry: dict[str, Any] | None) -> list[list[list[float]]]:
    if not geometry:
        return []

    coordinates = geometry.get("coordinates") or []
    geometry_type = geometry.get("type")
    if geometry_type == "Polygon":
        return [coordinates[0]] if coordinates else []
    if geometry_type == "MultiPolygon":
        return [polygon[0] for polygon in coordinates if polygon]
    return []


def _zip_property_value(properties: dict[str, Any]) -> str | None:
    for key in ["ZIP5", "zip", "zip_code", "ZIP_CODE", "postal_code"]:
        if key in properties:
            return normalize_zip(properties.get(key))
    return None


def _ring_center(ring: list[list[float]]) -> tuple[float, float]:
    x_values = [point[0] for point in ring]
    y_values = [point[1] for point in ring]
    return ((min(x_values) + max(x_values)) / 2.0, (min(y_values) + max(y_values)) / 2.0)


def plot_zip_level_choropleth(
    df: pd.DataFrame,
    *,
    zip_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    legend_label: str,
    label_zip_codes: list[str] | None = None,
    boundary_cache_path: Path = BOSTON_ZIP_BOUNDARY_CACHE_PATH,
    boundary_url: str = BOSTON_ZIP_BOUNDARY_URL,
) -> Path:
    """Render a static ZIP-level choropleth without requiring geopandas."""
    working = df.loc[:, [zip_col, value_col]].copy()
    working[zip_col] = working[zip_col].map(normalize_zip).astype("string")
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna(subset=[zip_col, value_col])
    if working.empty:
        raise ValueError("Cannot plot a ZIP choropleth without non-empty ZIP values.")

    zip_values = (
        working.groupby(zip_col, dropna=False)[value_col]
        .sum(min_count=1)
        .dropna()
        .to_dict()
    )
    if not zip_values:
        raise ValueError("Cannot plot a ZIP choropleth without numeric values.")

    boundary_geojson = load_boston_zip_boundaries(
        cache_path=boundary_cache_path,
        url=boundary_url,
    )

    patches: list[Polygon] = []
    patch_values: list[float] = []
    label_positions: dict[str, tuple[float, float]] = {}
    label_zip_set = (
        {normalize_zip(zip_code) for zip_code in label_zip_codes if normalize_zip(zip_code)}
        if label_zip_codes
        else None
    )

    for feature in boundary_geojson["features"]:
        properties = feature.get("properties") or {}
        zip_code = _zip_property_value(properties)
        rings = _outer_rings(feature.get("geometry"))
        if not rings:
            continue

        if zip_code is not None and zip_code in zip_values and zip_code not in label_positions:
            label_positions[zip_code] = _ring_center(rings[0])

        value = zip_values.get(zip_code, float("nan"))
        for ring in rings:
            patches.append(Polygon(ring, closed=True))
            patch_values.append(value)

    if not patches:
        raise ValueError("Boston ZIP boundary file did not include usable polygon geometry.")

    valid_values = [value for value in patch_values if pd.notna(value)]
    if not valid_values:
        raise ValueError("ZIP boundary map loaded, but none of the ZIP values matched the summary table.")

    min_value = min(valid_values)
    max_value = max(valid_values)
    if min_value == max_value:
        max_value = min_value + 1.0
    norm = Normalize(vmin=min_value, vmax=max_value)
    cmap = plt.get_cmap("YlOrRd")

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    facecolors = [
        "#e5e7eb" if pd.isna(value) else cmap(norm(value))
        for value in patch_values
    ]
    collection = PatchCollection(
        patches,
        facecolor=facecolors,
        edgecolor="white",
        linewidth=0.9,
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.72, pad=0.02)
    colorbar.set_label(legend_label)

    if label_zip_set is not None:
        for zip_code in label_zip_codes or []:
            normalized_zip = normalize_zip(zip_code)
            if normalized_zip not in label_positions:
                continue
            x_coord, y_coord = label_positions[normalized_zip]
            ax.text(
                x_coord,
                y_coord,
                normalized_zip,
                ha="center",
                va="center",
                fontsize=7,
                color="#111827",
            )
    elif len(label_positions) <= 30:
        for zip_code, (x_coord, y_coord) in label_positions.items():
            ax.text(
                x_coord,
                y_coord,
                zip_code,
                ha="center",
                va="center",
                fontsize=7,
                color="#111827",
            )

    ax.set_title(title)
    ax.text(
        0.01,
        0.01,
        "ZIPs without matching context are shown in light gray.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
    )
    return save_figure(output_path)
