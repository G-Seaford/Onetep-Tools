#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from ase import Atoms
from ase.io import read

from .data_structures import OnetepParams, GCeDFTParams
class OnetepCLI:
    """
    Encapsulates command-line parsing and application of CLI overrides
    to the strongly-typed ONETEP parameter packs.
    """

    # Construction / parsing
    def build_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            prog="generate_onetep_inputs.py",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=(
                "Generate ONETEP input (.dat) and SBATCH script(s) from ASE structures "
                "using modular, strongly-typed parameter packs."
            ),
        )

        # I/O paths and seed
        io = p.add_argument_group("I/O and job identity")
        io.add_argument("-i", "--input", type=Path, required=True, help="Input file or directory of structures (.xyz/.extxyz/.traj).",)
        io.add_argument("-o", "--output", type=Path, required=True,help="Output directory root; sub-folders created per structure/bias.",)
        io.add_argument("--seed", type=str, default=None,help="Job/file seed; defaults to structure stem (or stem-index for multi-frames).",)

        # Core numerics & physics
        core = p.add_argument_group("Core numerics and physics")
        core.add_argument("--task", type=str, default=None, help="ONETEP task, e.g. SinglePoint.")
        core.add_argument("--xc-functional", type=str, dest="xc", default=None, help="XC functional, e.g. OPTB88.")
        core.add_argument("--cutoff", type=int, default=None, help="Plane-wave equivalent cutoff (eV).")
        core.add_argument("--kgrid", type=self._csv_ints3, default=None, help="K-point grid as 'kx,ky,kz'.")
        core.add_argument("--kpar-groups", type=int, default=None, help="Number of k-par groups.")
        core.add_argument("--temperature", type=float, default=None, help="System temperature (K).")
        core.add_argument("--symmetry", dest="symmetry", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable symmetry AND time reversal together.",)
        core.add_argument("--use-paw", dest="use_paw", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable PAW.",)

        # Geometry (vacuum & axes)
        geom = p.add_argument_group("Geometry (vacuum and axes)")
        geom.add_argument("--vacuum", type=float, default=None, help="Vacuum thickness in Å to apply when centring the cell (use 0 to centre without extra space).",)
        geom.add_argument("--vacuum-axes", type=self._parse_axes, default=None, help="Axes to which vacuum is applied, e.g. '0,2' or 'x,z'. If omitted, all axes are used.",)
        geom.add_argument("--apply-vacuum", dest="apply_vacuum", action=argparse.BooleanOptionalAction, default=None,help="Enable/disable application of vacuum centring.",)

        # Geometry constraints
        gcons = p.add_argument_group("Geometry constraints")
        gcons.add_argument( "--use-constraints", dest="use_constraints", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable %%BLOCK SPECIES_CONSTRAINTS (effective for GeometryOptimization).",)
        gcons.add_argument( "--constraints", type=self._csv_constraints, default=None, help=("Species constraints mapping. Examples: " 
                                                                                             "'Au:FIXED'  or  'Au:LINE:1:0:0'  or  'C:PLANE:0:0:1'. "
                                                                                             "Multiple entries comma-separated, e.g. 'Au:LINE:1:0:0,C:FIXED'."
                                                                                             ),)
        gcons.add_argument( "--constraints-file", type=Path, default=None, help=("Path to a text file with one constraint per line"),)

        # Pseudopotentials
        pp = p.add_argument_group("Pseudopotentials")
        pp.add_argument("--pseudo-path", type=str, default=None, help="Directory containing pseudopotentials.")
        pp.add_argument("--pseudos", type=self._csv_map_str_str, default=None, help="Element:pseudo filename, e.g. 'C:C.PBE-paw.abinit,Au:Au.PBE-paw.abinit'.",)

        # eDFT / GC-eDFT
        ed = p.add_argument_group("eDFT / grand-canonical eDFT")
        ed.add_argument("--edft", dest="enable_edft", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable eDFT block.",)
        ed.add_argument("--gc-edft", dest="enable_gc_edft", action=argparse.BooleanOptionalAction, default=None,help="Enable/disable grand-canonical mode within eDFT.",)
        ed.add_argument("--edft-ref-electrode-potential", type=float, default=None, help="eDFT reference potential (eV).")
        ed.add_argument("--gc-applied-biases", type=self._csv_floats, default=None, help="Comma-separated applied potentials in V, e.g. '-0.1,0.0,0.1'.",)
        ed.add_argument("--spin", type=int, default=None, help="Total spin.")
        ed.add_argument("--spin-fix", type=int, default=None, help="eDFT spin fix (integer).")
        ed.add_argument("--spin-polarised", dest="spin_polarised", action=argparse.BooleanOptionalAction, default=None,help="Enable/disable spin polarisation.",)

        # Solvation & electrolyte
        solv = p.add_argument_group("Implicit solvation and PB electrolyte")
        solv.add_argument("--implicit-solvent", dest="enable_solvent", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable implicit solvent block.",)
        solv.add_argument("--pb-electrolyte", dest="enable_pbe", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Poisson–Boltzmann electrolyte within solvent.",)
        solv.add_argument("--implicit-ions", type=self._csv_implicit_ions, default=None, help="Implicit ions as 'H:+1:0.1,AuCl4:-1:0.1'. Only used if PB electrolyte is enabled.",)
        solv.add_argument("--solvation-radii", type=self._csv_map_str_float, default=None, help="Solvent radii (Bohr) as 'Au:3.14,C:3.20'.",)

        # NGWF
        ng = p.add_argument_group("NGWF")
        ng.add_argument("--ngwf-count", type=self._csv_map_str_int, default=None, help="Species→NGWF count, e.g. 'C:4,Au:6'.")
        ng.add_argument("--ngwf-radius", type=self._csv_map_str_float, default=None, help="Species→NGWF radius (Bohr), e.g. 'C:10.0,Au:12.0'.")

        # Output formatting
        fmt = p.add_argument_group("Output formatting")
        fmt.add_argument("--output-format", choices=("none", "dx", "cube", "both"), default=None,
            help=("Post-processing output choice. 'none' disables do_properties; 'dx' or 'cube' enables the respective format; 'both' enables both."),
        )

        # ONETEP executable root + binary name
        exe = p.add_argument_group("ONETEP executable/launcher")
        exe.add_argument("--onetep-root", type=Path, default=None, help="Root of an ONETEP build; binary expected in ROOT/bin and launcher in ROOT/utils.",)
        exe.add_argument("--binary-name", type=str, default=None,help="ONETEP binary name inside ROOT/bin (e.g. 'onetep.blythe_gnu').",)

        # Logging
        log = p.add_argument_group("Logging")
        log.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
        return p

    def parse(self, argv: list[str] | None) -> argparse.Namespace:
        return self.build_parser().parse_args(argv)

    # Apply CLI overrides
    def apply_cli_overrides(self, params: OnetepParams, args: argparse.Namespace) -> OnetepParams:
        core = params.core
        edft = params.edft
        io = params.io
        ng = params.ngwfs
        solv = params.solvent
        slurm = params.slurm
        geom = params.geom

        # Output root & seed
        core = replace(core, output_path=args.output)
        if args.seed is not None: core = replace(core, seed=args.seed)

        # Core
        if args.task is not None: core = replace(core, task=args.task)
        if args.xc is not None: core = replace(core, xc_functional=args.xc)
        if args.cutoff is not None: core = replace(core, cutoff_energy_ev=args.cutoff)
        if args.kgrid is not None: core = replace(core, kpoint_grid=args.kgrid)
        if args.kpar_groups is not None: core = replace(core, kpar_groups=args.kpar_groups)
        if args.temperature is not None: core = replace(core, temperature_k=args.temperature)
        if args.symmetry is not None: sym = bool(args.symmetry); core = replace(core, use_symmetry=sym, use_time_reversal=sym)
        if args.use_paw is not None: core = replace(core, use_paw=bool(args.use_paw))

        # Geometry
        if getattr(args, "vacuum", None) is not None: geom = replace(geom, vacuum=float(args.vacuum))
        if getattr(args, "vacuum_axes", None) is not None: geom = replace(geom, vacuum_axes=tuple(args.vacuum_axes))
        if getattr(args, "apply_vacuum", None) is not None: geom = replace(geom, apply_vacuum=bool(args.apply_vacuum))
        
        # Geometry constraints
        constraints_map = {}
        if getattr(args, "use_constraints", None) is not None: core = replace(core, geom_constraints=replace(core.geom_constraints, use_constraints=bool(args.use_constraints),),)
        if getattr(args, "constraints", None) is not None: constraints_map.update(args.constraints)
        if getattr(args, "constraints_file", None) is not None: constraints_map.update(self._load_constraints_file(args.constraints_file))
        if constraints_map: core = replace(core, geom_constraints=replace(core.geom_constraints, constraints=constraints_map,),)
        if (constraints_map or args.use_constraints) and args.task not in (None, "GeometryOptimization"): logging.warning("Species constraints provided but task is '%s'.", args.task,)
        # Pseudopotentials
        if args.pseudo_path is not None: core = replace(core, pseudo_path=args.pseudo_path)
        if getattr(args, "pseudos", None) is not None: core = replace(core, pseudopotentials=args.pseudos)

        # eDFT / GC eDFT
        if args.enable_edft is not None: edft = replace(edft, enable_edft=bool(args.enable_edft))
        if args.enable_gc_edft is not None: edft = replace(edft, enable_gc_edft=bool(args.enable_gc_edft))
        if edft.gc is None: edft = replace(edft, gc=GCeDFTParams())
        if args.edft_ref_electrode_potential is not None: edft = replace(edft, gc=replace(edft.gc, reference_potential_ev=args.edft_ref_electrode_potential))
        if args.gc_applied_biases is not None: edft = replace(edft, gc=replace(edft.gc, electrode_potentials_v=list(args.gc_applied_biases)))
        if args.spin is not None: edft = replace(edft, spin=args.spin)
        if args.spin_fix is not None: edft = replace(edft, spin_fix=args.spin_fix)
        if args.spin_polarised is not None: edft = replace(edft, spin_polarised=bool(args.spin_polarised))

        # Solvation / PB Electrolyte
        if args.enable_solvent is not None: solv = replace(solv, enable_implicit_solvent=bool(args.enable_solvent))
        if args.enable_pbe is not None: solv = replace(solv, electrolyte=replace(solv.electrolyte, enable_pbe=bool(args.enable_pbe)))
        if args.implicit_ions is not None: solv = replace(solv, electrolyte=replace(solv.electrolyte, implicit_ions=args.implicit_ions))
        if args.solvation_radii is not None: solv = replace(solv, solvent_radius_bohr=args.solvation_radii)

        # NGWF
        if args.ngwf_count is not None: ng = replace(ng, ngwfs_count=args.ngwf_count)
        if args.ngwf_radius is not None: ng = replace(ng, ngwfs_radius=args.ngwf_radius)

        # Output formatting policy
        if args.output_format is not None:
            if args.output_format == "none": io = replace(io, dx_format=False, cube_format=False); core = replace(core, write_outputs=False)
            elif args.output_format == "dx": io = replace(io, dx_format=True, cube_format=False); core = replace(core, write_outputs=True)
            elif args.output_format == "cube": io = replace(io, dx_format=False, cube_format=True); core = replace(core, write_outputs=True)
            elif args.output_format == "both": io = replace(io, dx_format=True, cube_format=True); core = replace(core, write_outputs=True)

        # Executable path override (root + binary name)
        if args.onetep_root is not None and args.binary_name is not None:
            bin_path = (args.onetep_root / "bin" / args.binary_name).as_posix()
            launcher_path = (args.onetep_root / "utils" / "onetep_launcher").as_posix()
            slurm = replace(slurm, onetep_binary=bin_path, onetep_launcher=launcher_path)
        elif (args.onetep_root is not None) ^ (args.binary_name is not None): logging.warning(
                "Provided only one of --onetep-root/--binary-name; ignoring both. "
                "Specify both to override SLURM executable paths."
            )

        return OnetepParams(core=core, io=io, ngwfs=ng, edft=edft, solvent=solv, geom=geom, slurm=slurm, extra=params.extra)

    # Structure discovery / reading
    def discover_structure_files(self, root: Path) -> list[Path]:
        """
        If root is a file, return [root]. If a directory, find .xyz/.extxyz/.traj (recursive).
        """
        if root.is_file(): return [root.resolve()]
        allowed = {".xyz", ".extxyz", ".traj"}
        files = sorted({p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed})
        return files

    def load_structures(self, path: Path) -> list[tuple[Atoms, str]]:
        """
        Read one or many Atoms from a file.
        Returns a list of (atoms, seed_suffix).
        """
        if path.suffix.lower() == ".traj": atoms_list: list[Atoms] = read(path, index=":"); return [(a, f"{path.stem}-{i:03d}") for i, a in enumerate(atoms_list)]
        else: a: Atoms = read(path); return [(a, path.stem)]

    # Static/utility parsers
    @staticmethod
    def _csv_floats(txt: str) ->list[float]:
        try:
            vals = [float(x) for x in txt.split(",") if x.strip() != ""]
            if not vals: raise ValueError("empty list")
            return vals
        except ValueError as exc: raise argparse.ArgumentTypeError(f"Could not parse floats from '{txt}'.") from exc

    @staticmethod
    def _csv_ints3(txt: str) -> tuple[int, int, int]:
        try: parts = [int(x) for x in txt.split(",")]
        except ValueError as exc: raise argparse.ArgumentTypeError("Expected integers in 'kx,ky,kz'.") from exc
        if len(parts) != 3: raise argparse.ArgumentTypeError("Expected exactly three integers: 'kx,ky,kz'.")
        return parts[0], parts[1], parts[2]

    @staticmethod
    def _csv_map_str_str(txt: str) -> dict[str, str]:
        out: dict[str, str] = {}
        if not txt: return out
        for item in txt.split(","):
            if ":" not in item: raise argparse.ArgumentTypeError( f"Expected NAME:VALUE entries in '{txt}', e.g. 'C:C.PBE...,Au:Au.PBE...'.")
            k, v = item.split(":", 1)
            out[k.strip()] = v.strip()
        if not out: raise argparse.ArgumentTypeError("Mapping cannot be empty.")
        return out

    @staticmethod
    def _csv_map_str_float(txt: str) -> dict[str, float]:
        raw = OnetepCLI._csv_map_str_str(txt)
        try: return {k: float(v) for k, v in raw.items()}
        except ValueError as exc: raise argparse.ArgumentTypeError(f"Expected floats in '{txt}'.") from exc

    @staticmethod
    def _csv_map_str_int(txt: str) -> dict[str, int]:
        raw = OnetepCLI._csv_map_str_str(txt)
        try: return {k: int(v) for k, v in raw.items()}
        except ValueError as exc: raise argparse.ArgumentTypeError(f"Expected integers in '{txt}'.") from exc

    @staticmethod
    def _csv_implicit_ions(txt: str) -> list[dict[str, str | int | float]]:
        ions: list[dict[str, str | int | float]] = []
        for item in txt.split(","):
            parts = [p.strip() for p in item.split(":")]
            if len(parts) != 3: raise argparse.ArgumentTypeError( "Implicit ions must be 'SYMBOL:CHARGE:CONC' separated by commas." )
            
            sym = parts[0]
            try: charge = int(parts[1]); conc = float(parts[2])
            except ValueError as exc: raise argparse.ArgumentTypeError(f"Bad implicit ion entry '{item}' (charge must be int, conc a float).") from exc
            
            ions.append({"symbol": sym, "charge": charge, "conc": conc})
        if not ions: raise argparse.ArgumentTypeError("Implicit ions list cannot be empty.")
        return ions

    @staticmethod
    def _parse_axes(txt: str) -> tuple[int, ...]:
        txt = txt.strip().lower()
        tokens = [t.strip() for t in (txt.split(",") if "," in txt else list(txt)) if t.strip()]
        axes: list[int] = []
        for t in tokens:
            if t in {"0", "1", "2"}: axes.append(int(t))
            elif t in {"x", "y", "z"}: axes.append({"x": 0, "y": 1, "z": 2}[t])
            else: raise argparse.ArgumentTypeError(f"Bad axis token '{t}'. Use 0/1/2 or x/y/z (comma-separated).")
            
        seen: set[int] = set()
        dedup: list[int] = []
        for a in axes:
            if a not in {0, 1, 2}: raise argparse.ArgumentTypeError("Axis index must be 0, 1 or 2.")
            if a not in seen: dedup.append(a); seen.add(a)
        if not dedup: raise argparse.ArgumentTypeError("No valid axes provided.")
        return tuple(dedup)
    
    @staticmethod
    def _csv_constraints(txt: str) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for item in (x.strip() for x in txt.split(",") if x.strip()):
            parts = [p.strip() for p in item.split(":")]
            if len(parts) < 2: raise argparse.ArgumentTypeError(f"Bad constraint entry '{item}'. Expected 'SPEC:TYPE[:x:y:z]'.")
            spec = parts[0]
            
            ctype = parts[1].upper()
            if ctype in {"NONE", "FIXED"}:
                if len(parts) != 2: raise argparse.ArgumentTypeError(f"'{item}': {ctype} takes no vector.")
                mapping[spec] = ctype
            elif ctype in {"LINE", "PLANE"}:
                if len(parts) != 5: raise argparse.ArgumentTypeError(f"'{item}': {ctype} requires x:y:z.")
                try: x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError as exc: raise argparse.ArgumentTypeError(f"'{item}': x,y,z must be numbers.") from exc
                mapping[spec] = f"{ctype} {x} {y} {z}"
            else: raise argparse.ArgumentTypeError(f"'{item}': unknown type '{ctype}' (use NONE, FIXED, LINE, PLANE).")
        if not mapping: raise argparse.ArgumentTypeError("Constraints mapping cannot be empty.")
        return mapping

    @staticmethod
    def _load_constraints_file(path: Path) -> dict[str, str]:
        
        if not path.exists(): raise argparse.ArgumentTypeError(f"Constraints file not found: {path}")
        
        mapping: dict[str, str] = {}
        text = path.read_text().splitlines()
        for line in text:
            s = line.split("#", 1)[0].strip()
            if not s: continue
            
            # Accept forms: "Au LINE 1 0 0" or "C FIXED"
            parts = s.split()
            if len(parts) == 2:
                spec, ctype = parts[0], parts[1].upper()
                if ctype not in {"NONE", "FIXED"}: raise argparse.ArgumentTypeError(f"Bad line '{line}': expected NONE or FIXED.")
                
                mapping[spec] = ctype
            elif len(parts) == 5:
                spec, ctype = parts[0], parts[1].upper()
                if ctype not in {"LINE", "PLANE"}:
                    raise argparse.ArgumentTypeError(f"Bad line '{line}': expected LINE or PLANE.")
                try: x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError as exc: raise argparse.ArgumentTypeError(f"Bad line '{line}': x y z must be numbers.") from exc
                mapping[spec] = f"{ctype} {x} {y} {z}"
            else: raise argparse.ArgumentTypeError( f"Bad line '{line}': expected 'SPEC TYPE' or 'SPEC TYPE x y z'.")
            
        return mapping