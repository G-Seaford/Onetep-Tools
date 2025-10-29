#!/usr/bin/env python3
from __future__ import annotations

import logging, sys
from dataclasses import replace, is_dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

from ase import Atoms
from ase.io import write

from onetep_tools.data_structures import OnetepParams
from onetep_tools.cli_tools import OnetepCLI
from onetep_tools.bash_template import write_sbatch

from onetep_keywords import DEFAULT_PARAMETERS

T = TypeVar("T")

# Logging
def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1: level = logging.INFO
    elif verbosity >= 2: level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

# Helpers
def species_in_atoms(atoms: Atoms) -> set[str]:
    return set(atoms.get_chemical_symbols())

def filter_map_for_species(mapping: dict[str, T], species: Iterable[str], *, label: str) -> dict[str, T]:
    s = set(species)
    filtered: dict[str, T] = {k: v for k, v in mapping.items() if k in s}
    dropped = set(mapping) - s
    if dropped:
        logging.debug("Dropping %s entries not present in structure: %s", label, ", ".join(sorted(dropped)))
    return filtered

def check_required_species(mapping: dict[str, object], species: set[str], *, what: str) -> None:
    missing = sorted([sp for sp in species if sp not in mapping])
    if missing:
        logging.error("Missing %s for species: %s", what, ", ".join(missing))

def fix_keywords_solvent_radii(keywords: dict[str, object], species: set[str]) -> None:
    """
    Trim ONETEP keyword 'species_solvent_radius' (list[str] entries like 'Au 3.14')
    so it only includes species present in the current structure.
    """
    key = "species_solvent_radius"
    if key not in keywords: return

    pairs = keywords.get(key)
    if not isinstance(pairs, list): return

    filtered: list[str] = []
    for entry in pairs:
        if isinstance(entry, str) and entry:
            name = entry.split()[0]
            if name in species: filtered.append(entry)

    if not filtered and pairs: logging.warning("All solvent radii entries were for absent species; removing key from keywords."); keywords.pop(key, None)
    else: keywords[key] = filtered

def format_bias_dir(pot: float | None) -> str|None:
    return f"{pot:+.2f}V" if pot is not None else None

def generate_for_file(params: OnetepParams, in_file: Path, cli: OnetepCLI, single_file:bool) -> None:
    structs = cli.load_structures(in_file)
    if not structs: logging.warning("No structures found in '%s'.", in_file); return
    
    # Determine where to write outputs
    if single_file:
        file_output_root = params.core.output_path
        logging.info("Single file mode: writing directly under %s", file_output_root)
        
    else:
        base_name = in_file.stem
        file_output_root = params.core.output_path / base_name
        file_output_root.mkdir(parents=True, exist_ok=True)
        logging.info("Output for %s will be written under: %s", in_file.name, file_output_root)

    potentials = params.iter_applied_potentials()
    logging.debug("Bias sweep: %s", potentials)
    
    single_bias_mode = (len(potentials) == 1)
    if single_bias_mode: 
        potential = potentials[0]
        if potential is None: logging.info("Single-bias mode: neutral (no applied bias)")
        else: logging.info("Single-bias mode: %.2f V", potential)

    for atoms, default_seed in structs:
        sp = species_in_atoms(atoms)
        
        atoms.set_pbc(params.geom.pbc)
        axes = params.geom.vacuum_axes if params.geom.vacuum_axes is not None else (2)
        if params.geom.apply_vacuum: logging.debug("Applying vacuum = %.3f Å on axes %s", params.geom.vacuum, axes); atoms.center(vacuum=params.geom.vacuum, axis=axes)
        else: logging.debug("Recentring on axes %s", axes); atoms.center(vacuum=0.0, axis=axes)

        # Per-structure filtered dictionaries
        filtered_pseudos = filter_map_for_species(params.core.pseudopotentials, sp, label="pseudopotentials")
        filtered_ngwf_count = filter_map_for_species(params.ngwfs.ngwfs_count, sp, label="NGWF counts")
        filtered_ngwf_radius = filter_map_for_species(params.ngwfs.ngwfs_radius, sp, label="NGWF radii")
        filtered_solvent_radii = filter_map_for_species(params.solvent.solvent_radius_bohr, sp, label="solvation radii")

        # Warn if essentials are missing
        check_required_species(filtered_pseudos, sp, what="pseudopotentials")
        check_required_species(filtered_ngwf_count, sp, what="NGWF counts")
        check_required_species(filtered_ngwf_radius, sp, what="NGWF radii")

        # Build a shallow, per-structure copy with filtered maps for keyword generation
        params_local = replace(
            params,
            core=replace(params.core, pseudopotentials=filtered_pseudos),
            ngwfs=replace(params.ngwfs, ngwfs_count=filtered_ngwf_count, ngwfs_radius=filtered_ngwf_radius),
            solvent=replace(params.solvent, solvent_radius_bohr=filtered_solvent_radii),
        )

        seed = params.core.seed or default_seed

        for pot in potentials:

            if single_bias_mode: subdir = file_output_root
            else: 
                label = format_bias_dir(pot) or "0.00V"
                subdir = file_output_root / label
            
            subdir.mkdir(parents=True, exist_ok=True)

            keywords = params_local.build_keywords(pot)
            fix_keywords_solvent_radii(keywords, sp)

            dat_path = subdir / f"{seed}.dat"
            logging.info("Writing %s", dat_path)
            logging.debug("Keywords count: %d", len(keywords))

            write(
                dat_path,
                atoms,
                format="onetep-in",
                keywords=keywords,
                pseudo_path=params_local.core.pseudo_path,
                pseudopotentials=params_local.core.pseudopotentials,
                ngwf_count=params_local.ngwfs.ngwfs_count,
                ngwf_radius=params_local.ngwfs.ngwfs_radius,
            )

            write_sbatch(subdir, seed, params_local.slurm)


def _deep_replace(dc, overrides: dict[str, Any]):
    """
    Recursively apply a nested dict of overrides onto a nested dataclass tree.
    Unknown keys are ignored (debug-logged). Dict fields (e.g. pseudopotentials)
    are assigned directly; nested dataclasses are recursed with replace().
    """
    updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(dc, key): logging.debug("Ignoring unknown override key '%s' for %s", key, type(dc).__name__); continue
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict): updates[key] = _deep_replace(current, value)
        else: updates[key] = value
    return replace(dc, **updates)

# Main Routine
def main(argv: list[str] | None = None) -> int:
    cli = OnetepCLI()
    args = cli.parse(argv)

    setup_logging(args.verbose)

    input_path: Path = args.input
    if not input_path.exists(): logging.error("Input path does not exist: %s", input_path); return 2

    files = cli.discover_structure_files(input_path)
    if not files: logging.error("No input structures found under '%s' (expected .xyz/.extxyz/.traj).", input_path); return 2

    params = OnetepParams()
    params = _deep_replace(params, DEFAULT_PARAMETERS) # Set default parameters from the `onetep_inputs.py` file
    params = cli.apply_cli_overrides(params, args) # Apply command-line overrides

    # Create output root
    params.core.output_path.mkdir(parents=True, exist_ok=True)

    n_files = len(files)
    logging.info("Found %d input file(s).", n_files)
    for f in files:
        logging.info("Processing %s", f)
        try: generate_for_file(params, f, cli, single_file=(n_files == 1))
        except Exception as exc: logging.exception("Failed to generate inputs for '%s': %s", f, exc)

    logging.info("All done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
