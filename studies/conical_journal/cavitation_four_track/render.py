#!/usr/bin/env pvpython
"""Render selected OpenFOAM multiphase states with a fixed camera and scale."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

from paraview.simple import (  # type: ignore[import-not-found]
    CellDatatoPointData,
    ColorBy,
    CreateView,
    ExtractSurface,
    GetScalarBar,
    OpenFOAMReader,
    SaveScreenshot,
    Show,
    Text,
    Threshold,
)
from vtkmodules.vtkCommonCore import vtkVersion

from bearing_cfd.artifacts import make_staging_directory, publish_generation


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", required=True, help="comma-separated OpenFOAM times")
    parser.add_argument("--field", choices=("alpha.oil", "p_rgh"), required=True)
    parser.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"), required=True)
    parser.add_argument("--threshold", type=float, help="render only cells at or below this value")
    parser.add_argument("--title", required=True)
    parser.add_argument("--status", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(values)
    case = args.case.resolve()
    target = args.output.resolve()
    times = [float(value) for value in args.times.split(",")]
    foam = next(case.glob("*.foam"), None)
    if foam is None:
        raise FileNotFoundError(f"no .foam marker found in {case}")
    output = make_staging_directory(target)
    try:
        _render(args, case, output, foam, times)
        mesh_files = tuple(
            case / "constant/polyMesh" / name
            for name in ("boundary", "faces", "neighbour", "owner", "points")
        )
        publish_generation(
            output,
            target,
            stage="study",
            operation="cavitation-four-track-render",
            status="RENDERED",
            resolved_inputs={
                "case": case,
                "times": times,
                "field": args.field,
                "range": args.range,
                "threshold": args.threshold,
                "title": args.title,
                "status": args.status,
            },
            input_units={"pressure": "Pa absolute", "alpha.oil": "dimensionless"},
            producer_files=(Path(__file__),),
            upstream_artifacts=(
                foam,
                *(case / f"{time:g}" / args.field for time in times),
                *mesh_files,
            ),
            tool_versions={"vtk": vtkVersion.GetVTKVersion()},
            argv=values,
            acceptance_status="EVIDENCE_ONLY",
            repository=REPO_ROOT,
        )
    finally:
        if output.exists():
            shutil.rmtree(output)
    print(f"wrote multiphase renders to {target}")
    return 0


def _render(
    args: argparse.Namespace,
    case: Path,
    output: Path,
    foam: Path,
    times: list[float],
) -> None:
    reader = OpenFOAMReader(FileName=str(foam.resolve()))
    reader.MeshRegions = ["internalMesh"]
    reader.CellArrays = ["alpha.oil", "p_rgh"]
    source = reader
    if args.threshold is not None:
        source = Threshold(Input=reader)
        source.Scalars = ["CELLS", args.field]
        source.LowerThreshold = args.range[0]
        source.UpperThreshold = args.threshold
    surface = ExtractSurface(Input=source)
    point_surface = CellDatatoPointData(Input=surface)

    view = CreateView("RenderView")
    view.ViewSize = [1280, 900]
    view.UseColorPaletteForBackground = 0
    view.Background = [0.96, 0.96, 0.96]
    view.OrientationAxesVisibility = 1
    view.CameraPosition = [0.145, -0.165, 0.135]
    view.CameraFocalPoint = [0, 0, 0.05]
    view.CameraViewUp = [0, 0, 1]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 0.088

    if args.threshold is not None:
        context_surface = ExtractSurface(Input=reader)
        context_display = Show(context_surface, view)
        context_display.Representation = "Surface"
        context_display.AmbientColor = [0.65, 0.67, 0.70]
        context_display.DiffuseColor = [0.65, 0.67, 0.70]
        context_display.Opacity = 0.13

    display = Show(point_surface, view)
    display.Representation = "Surface"
    display.Ambient = 1
    display.Diffuse = 0
    display.Specular = 0
    ColorBy(display, ("POINTS", args.field))
    lut = display.LookupTable
    lut.ColorSpace = "RGB"
    midpoint = sum(args.range) / 2
    lut.RGBPoints = [
        args.range[0], 0.267, 0.005, 0.329,
        midpoint, 0.128, 0.567, 0.551,
        args.range[1], 0.993, 0.906, 0.144,
    ]
    lut.RescaleTransferFunction(*args.range)

    scalar_bar = GetScalarBar(lut, view)
    scalar_bar.Title = "oil volume fraction" if args.field == "alpha.oil" else "absolute pressure [Pa]"
    scalar_bar.ComponentTitle = ""
    scalar_bar.TitleColor = [0, 0, 0]
    scalar_bar.LabelColor = [0, 0, 0]
    scalar_bar.TitleFontSize = 18
    scalar_bar.LabelFontSize = 15
    scalar_bar.ScalarBarLength = 0.62
    scalar_bar.WindowLocation = "Any Location"
    scalar_bar.Position = [0.84, 0.18]
    display.SetScalarBarVisibility(view, True)

    heading = Text()
    heading.Text = args.title
    heading_display = Show(heading, view)
    heading_display.WindowLocation = "Upper Center"
    heading_display.Color = [0.05, 0.05, 0.05]
    heading_display.FontSize = 19

    status = Text()
    status.Text = args.status
    status_display = Show(status, view)
    status_display.WindowLocation = "Lower Left Corner"
    status_display.Color = [0.72, 0.04, 0.04]
    status_display.FontSize = 15

    for index, time_value in enumerate(times):
        view.ViewTime = time_value
        reader.UpdatePipeline(time_value)
        if args.threshold is not None:
            source.UpdatePipeline(time_value)
        surface.UpdatePipeline(time_value)
        point_surface.UpdatePipeline(time_value)
        heading.Text = f"{args.title}\nOpenFOAM time = {time_value:g}"
        SaveScreenshot(
            str(output / f"{args.field.replace('.', '_')}_{index:03d}.png"),
            view,
            ImageResolution=[1280, 900],
        )


if __name__ == "__main__":
    raise SystemExit(main())
