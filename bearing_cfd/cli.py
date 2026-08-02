"""Installed command-line interface for bearing-cfd."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


OUTPUT_ROOT = Path("out/conical_journal")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bearing-cfd", add_help=False)
    parser.add_argument("bearing", choices=("conical-journal",))
    parser.add_argument(
        "stage", choices=("geometry", "mesh", "export", "simulate", "study")
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def _defaults(argv: Sequence[str], **options: str | Path) -> list[str]:
    result = list(argv)
    for option, value in reversed(tuple(options.items())):
        flag = "--" + option.replace("_", "-")
        if not _has_option(result, flag):
            result[0:0] = [flag, str(value)]
    return result


def _top_help() -> str:
    return """usage: bearing-cfd conical-journal <stage> [operation] [options]

stages:
  geometry
  mesh      brep-preflight | surface-smoke | no-port | surface-inlet |
            body-fitted-inlet | central-feed
  export    central-feed
  simulate  reynolds-jfo
  study     body-fitted-mesh | jfo-feed-geometry | jfo-checkpoint-evidence |
            hydrodynamic-ramp | single-phase-oq90 | cavitation-four-track
"""


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values in (["-h"], ["--help"]):
        print(_top_help())
        return 0
    namespace = _parser().parse_args(values)
    remaining = list(namespace.arguments)
    if namespace.bearing != "conical-journal":  # argparse currently prevents this
        _parser().error("unsupported bearing")

    if namespace.stage == "geometry":
        from bearing_cfd.bearings.conical_journal.geometry import cad

        return cad.main(
            _defaults(remaining, outdir=OUTPUT_ROOT / "geometry/default")
        )

    if not remaining:
        _parser().error(f"{namespace.stage} requires an operation")
    operation = remaining.pop(0)

    if namespace.stage == "mesh":
        geometry = OUTPUT_ROOT / "geometry/default"
        preflight = OUTPUT_ROOT / "meshing/brep-preflight"
        if operation == "brep-preflight":
            from bearing_cfd.bearings.conical_journal.meshing import brep_preflight

            return brep_preflight.main(
                _defaults(
                    remaining,
                    unsplit=geometry / "film_unsplit.brep",
                    zones=geometry / "film_zones.brep",
                    params=geometry / "params.json",
                    outdir=preflight,
                )
            )
        if operation == "surface-smoke":
            from bearing_cfd.bearings.conical_journal.meshing import surface_smoke

            return surface_smoke.main(
                _defaults(
                    remaining,
                    brep=preflight / "film_zones_fragmented.brep",
                    preflight_report=preflight / "preflight_report.json",
                    outdir=OUTPUT_ROOT / "meshing/surface-smoke",
                )
            )
        if operation == "no-port":
            from bearing_cfd.bearings.conical_journal.meshing import no_port

            return no_port.main(
                _defaults(
                    remaining,
                    params=geometry / "params.json",
                    outdir=OUTPUT_ROOT / "meshing/no-port",
                )
            )
        if operation == "surface-inlet":
            from bearing_cfd.bearings.conical_journal.meshing import surface_inlet

            return surface_inlet.main(
                _defaults(
                    remaining,
                    params=geometry / "params.json",
                    outdir=OUTPUT_ROOT / "meshing/surface-inlet",
                )
            )
        if operation == "body-fitted-inlet":
            from bearing_cfd.bearings.conical_journal.meshing import body_fitted_inlet

            return body_fitted_inlet.main(
                _defaults(
                    remaining,
                    params=geometry / "params.json",
                    outdir=OUTPUT_ROOT / "meshing/body-fitted-inlet",
                )
            )
        if operation == "central-feed":
            from bearing_cfd.bearings.conical_journal.meshing import central_feed

            return central_feed.main(
                _defaults(
                    remaining,
                    params=geometry / "params.json",
                    brep=geometry / "film_unsplit.brep",
                    preflight=preflight / "preflight_report.json",
                    outdir=OUTPUT_ROOT / "meshing/central-feed",
                )
            )
        _parser().error(f"unknown mesh method: {operation}")

    if namespace.stage == "export" and operation == "central-feed":
        from bearing_cfd.bearings.conical_journal import interchange

        return interchange.main(
            _defaults(
                remaining,
                case_dir=OUTPUT_ROOT / "meshing/central-feed/nGap_08",
                outdir=OUTPUT_ROOT / "interchange/central-feed",
            )
        )

    if namespace.stage == "simulate" and operation == "reynolds-jfo":
        from bearing_cfd.bearings.conical_journal.simulation import reynolds_jfo

        return reynolds_jfo.main(
            _defaults(remaining, outdir=OUTPUT_ROOT / "simulation/reynolds-jfo")
        )

    if namespace.stage == "study":
        outdir = OUTPUT_ROOT / "studies" / operation
        if operation == "body-fitted-mesh":
            from studies.conical_journal.body_fitted_mesh import study

            return study.main(_defaults(remaining, outdir=outdir))
        if operation == "jfo-feed-geometry":
            from studies.conical_journal.jfo_feed_geometry import render

            return render.main(_defaults(remaining, outdir=outdir)) or 0
        if operation == "jfo-checkpoint-evidence":
            from studies.conical_journal.jfo_checkpoint_evidence import render

            return render.main(_defaults(remaining, output=outdir))
        if operation == "hydrodynamic-ramp":
            from studies.conical_journal.hydrodynamic_ramp import run

            return run.main(_defaults(remaining, work_case=outdir))
        if operation == "single-phase-oq90":
            from studies.conical_journal import single_phase_oq90

            pvpython = shutil.which("pvpython")
            if pvpython is None:
                print("pvpython is required for this study", file=sys.stderr)
                return 2
            script = Path(single_phase_oq90.__file__).with_name("render.py")
            environment = os.environ.copy()
            package_root = Path(__file__).resolve().parents[1]
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (str(package_root), environment.get("PYTHONPATH", "")),
                )
            )
            return subprocess.run(
                [pvpython, str(script), *_defaults(remaining, output=outdir)],
                env=environment,
                check=False,
            ).returncode
        if operation == "cavitation-four-track":
            from studies.conical_journal.cavitation_four_track import summarize

            return summarize.main(_defaults(remaining, output=outdir)) or 0

    _parser().error(f"unsupported command: {namespace.stage} {operation}")
    return 2
