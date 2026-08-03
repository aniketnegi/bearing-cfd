"""Installed command-line interface for bearing-cfd."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


OUTPUT_ROOT = Path("out/conical_journal")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bearing-cfd", add_help=False)
    parser.add_argument("bearing", choices=("conical-journal",))
    parser.add_argument(
        "stage", choices=("geometry", "mesh", "export", "simulate")
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

    _parser().error(f"unsupported command: {namespace.stage} {operation}")
    return 2
