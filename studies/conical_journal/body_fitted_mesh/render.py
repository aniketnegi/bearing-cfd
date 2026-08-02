#!/usr/bin/env pvpython
"""Render one mesh preview with ParaView's off-screen renderer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from paraview.simple import (  # type: ignore[import-not-found]
    ColorBy,
    CreateView,
    GetColorTransferFunction,
    OpenDataFile,
    ResetCamera,
    SaveScreenshot,
    Show,
    Text,
    UpdatePipeline,
    _DisableFirstRenderCameraReset,
)


def render(
    source: Path,
    output: Path,
    *,
    title: str,
    rejected: bool,
) -> None:
    _DisableFirstRenderCameraReset()
    reader = OpenDataFile(str(source.resolve()))
    if reader is None:
        raise RuntimeError(f"ParaView could not read {source}")
    UpdatePipeline(proxy=reader)
    view = CreateView("RenderView")
    view.ViewSize = [1200, 820]
    view.UseColorPaletteForBackground = 0
    view.Background = [0.96, 0.97, 0.99]
    view.OrientationAxesVisibility = 1
    display = Show(reader, view)
    display.Representation = "Surface With Edges"
    display.AmbientColor = [0.20, 0.42, 0.72]
    display.DiffuseColor = [0.35, 0.62, 0.92]
    display.EdgeColor = [0.08, 0.11, 0.16]
    display.LineWidth = 0.35

    cell_arrays = set(reader.CellData.keys())
    colour_field = next(
        (
            name
            for name in ("pressure_feed", "patch_id", "block_id")
            if name in cell_arrays
        ),
        None,
    )
    if colour_field is not None:
        ColorBy(display, ("CELLS", colour_field))
        display.RescaleTransferFunctionToDataRange(True, False)
        if colour_field == "pressure_feed":
            lookup = GetColorTransferFunction(colour_field)
            lookup.ColorSpace = "RGB"
            lookup.RGBPoints = [
                0.0,
                0.68,
                0.78,
                0.90,
                1.0,
                0.95,
                0.24,
                0.10,
            ]
        display.SetScalarBarVisibility(view, False)

    ResetCamera(view)
    bounds = reader.GetDataInformation().GetBounds()
    centre = [
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ]
    span = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )
    view.CameraFocalPoint = centre
    view.CameraPosition = [
        centre[0] + 1.7 * span,
        centre[1] + 1.9 * span,
        centre[2] + 1.25 * span,
    ]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 0.78 * span

    label = Text()
    label.Text = (
        f"{title}\nREJECTED — VISUAL ONLY — DO NOT SOLVE"
        if rejected
        else f"{title}\nPASSED STATIC MESH GATES"
    )
    label_display = Show(label, view)
    label_display.WindowLocation = "Upper Left Corner"
    label_display.FontSize = 20
    label_display.Color = [0.78, 0.08, 0.08] if rejected else [0.03, 0.42, 0.20]

    output.parent.mkdir(parents=True, exist_ok=True)
    SaveScreenshot(
        str(output.resolve()),
        view,
        ImageResolution=[1200, 820],
        TransparentBackground=0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--rejected", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    render(
        arguments.input,
        arguments.output,
        title=arguments.title,
        rejected=arguments.rejected,
    )
