#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import ChainMap

from typing import Any, Optional
from ase.calculators.onetep import OnetepProfile

# Typing aliases
KeywordValue = str | int | float | bool | list[str]
KeywordDict = dict[str, KeywordValue]

# Sentinel for unset values
class _UnsetType: pass
UNSET = _UnsetType()

# Helpers
def _strip_nones(d: dict[str, Any]) -> KeywordDict:
    return {k: v for k, v in d.items() if v is not None}

def _merge_keywords(*dicts: dict[str, Any]) -> KeywordDict:
    dicts = [d for d in dicts if d]
    merged = dict(ChainMap(*reversed(list(dicts))))
    return _strip_nones(merged)

def _opt(v: float | _UnsetType | None, unit: str) -> str | None:
    return None if (v is None or isinstance(v, _UnsetType)) else f"{v} {unit}"

# Base class
@dataclass
class ParamsTemplate:
    """
    Template for ONETEP parameter classes.

    Subclasses:
      - set `enable_attr` to the name of the boolean variable controlling usage (or None if always on)
      - override `_core_keywords(ctx)` to return ONETEP-ready keys (no internal names)
      - put ad-hoc overrides into `extra`; they are merged last.
    """
    enable_attr: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Context:
        temperature_k: float | _UnsetType = UNSET
        applied_potential_v: float | _UnsetType = UNSET

    def _is_enabled(self) -> bool:
        if self.enable_attr is None: return True
        return bool(getattr(self, self.enable_attr))

    def _core_keywords(self, ctx: Context) -> dict[str, Any]:
        """Override in subclasses."""
        return {}

    def to_keywords(self, ctx: Context) -> KeywordDict:
        if not self._is_enabled(): return {}
        base = self._core_keywords(ctx)
        return _merge_keywords(base, self.extra)

# Dataclasses
@dataclass
class GeometryParams:
    """Non-keyword geometry tweaks applied to Atoms prior to writing."""
    pbc: tuple[bool, bool, bool] = (True, True, False)
    apply_vacuum: bool = True
    vacuum: float|None = 20.0
    vacuum_axes: tuple[int, ...]|None = (2,)
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class OnetepIO(ParamsTemplate):
    """ONETEP I/O and formatting (no write_forces here by design)."""
    output_detail: str = "verbose"
    
    # Output formats
    dx_format: bool = True
    cube_format: bool = False

    # Write options
    write_xyz: bool = True
    write_hamiltonian: bool = True
    write_denskern: bool = False
    
    # Read options
    read_tightbox_ngwfs: bool = False
    read_hamiltonian: bool = False
    read_denskern: bool = False

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        return {
            "output_detail": self.output_detail,
            "forces_output_detail": self.output_detail,
            "dx_format": self.dx_format,
            "cube_format": self.cube_format,
            "write_xyz": self.write_xyz,
            "write_hamiltonian": self.write_hamiltonian,
            "write_denskern": self.write_denskern,
            "read_tightbox_ngwfs": self.read_tightbox_ngwfs,
            "read_hamiltonian": self.read_hamiltonian,
            "read_denskern": self.read_denskern,
        }

@dataclass
class GeomConstraints:
    """
    Internal geometry constraints for ONETEP.
    Dict maps species to a constraint string,
    e.g. {"Au": "LINE 1.0 0.0 0.0", "C": "FIXED"}.
    """
    use_constraints: bool = False
    constraints: dict[str, str] = field(default_factory=dict)

    # Allowed constraint types
    _ALLOWED_TYPES: tuple[str, ...] = ("NONE", "FIXED", "LINE", "PLANE")

    def _format_constraint(self, spec: str) -> Optional[str]:
        """
        Validate and normalise a constraint string.
        Accepts:
          "NONE", "FIXED", "LINE 1 0 0", "PLANE 0 0 1"
        Returns canonical form or None if invalid.
        """
        if not spec or not spec.strip(): return None
        
        # Strip trailing comments
        spec = spec.split(";", 1)[0].strip()
        parts = spec.split()
        if not parts: return None

        ctype = parts[0].upper()
        if ctype not in self._ALLOWED_TYPES: return None

        if ctype in {"NONE", "FIXED"}:
            if len(parts) != 1: return None
            return ctype

        if len(parts) != 4: return None
        try:  x, y, z = map(float, parts[1:])
        except ValueError: return None
        return f"{ctype} {x} {y} {z}"

    def species_constraints(self, *, species: set[str] | None = None) -> list[str]:
        """
        Return a list of constraint lines (one per species),
        suitable for ASE's list→%BLOCK handling.
        e.g. ["Au LINE 1.0 0.0 0.0", "C FIXED"]
        """
        if not self.use_constraints or not self.constraints: return []

        lines: list[str] = []
        for sp in sorted(self.constraints):
            if species is not None and sp not in species: continue

            norm = self._format_constraint(self.constraints[sp])
            if not norm: continue
            
            lines.append(f"{sp} {norm}")

        return lines

@dataclass
class CoreParams(ParamsTemplate):
    """Core ONETEP run parameters and paths."""
    
    # Basic settings
    task: str = "SinglePoint"
    xc_functional: str = "OPTB88"
    kpoint_method: str = "PW"
    use_paw: bool = True
    write_forces: bool = True
    write_outputs: bool = True

    # Paths & IO
    pseudo_path: str = "/springbrook/home/p/phumhp/Au-Graphene/pseudos"
    pseudopotentials: dict[str, str] = field( default_factory=lambda: {"C": "C.PBE-paw.abinit", "Au": "Au.PBE-paw.abinit"})

    input_path: Path = Path("Structures/Clusters/2025-08-31")
    output_path: Path = Path("Calculations/Clusters")
    seed: str | None = None  # file root to use for .dat/.out/.err

    # System parameters
    cutoff_energy_ev: int = 750
    kpoint_grid: tuple[int, int, int] = (4, 4, 1)
    kpar_groups: int = 3
    temperature_k: float = 298.15
    fine_grid_scale: float = 4.0
    
    # Symmetry Operations
    use_symmetry: bool = True
    use_time_reversal: bool = True
    
    # Geometry constraints
    geom_constraints: GeomConstraints = field(default_factory=GeomConstraints)

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        g = self.kpoint_grid
        return {
            "task": self.task,
            "cutoff_energy": f"{self.cutoff_energy_ev} eV",
            "xc_functional": self.xc_functional,
            "paw": self.use_paw,
            "write_forces": self.write_forces,
            'do_properties' : self.write_outputs,
            "kpoint_method": self.kpoint_method,
            "kpoint_grid_size": f"{g[0]} {g[1]} {g[2]}",
            "num_kpars": self.kpar_groups,
            "use_symmetry": self.use_symmetry,
            "use_time_reversal": self.use_time_reversal,
            "fine_grid_scale": self.fine_grid_scale,
        }

@dataclass
class NGWFParams(ParamsTemplate):
    """NGWF settings."""
    extend_ngwfs: tuple[bool, bool, bool] = (True, True, True)
    ngwfs_count: dict[str, int] = field(default_factory=lambda: {"C": 4, "Au": 6})
    ngwfs_radius: dict[str, float] = field(default_factory=lambda: {"C": 10.0, "Au": 12.0})
    write_tightbox_ngwfs: bool = True

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        e = self.extend_ngwfs
        return {
            "extend_ngwf": f"{e[0]} {e[1]} {e[2]}",
            "write_tightbox_ngwfs": self.write_tightbox_ngwfs,
        }

@dataclass
class GCeDFTParams(ParamsTemplate):
    
    reference_potential_ev: float = -2.70
    electrode_potentials_v: list[float] = field(default_factory=lambda: [-0.10, 0.00, 0.10])

    @property
    def electrode_potentials(self) -> list[float]:
        """Read-only-ish view (copy) of the sweep as a list."""
        return list(self.electrode_potentials_v)

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        return {
            "edft_reference_potential": f"{self.reference_potential_ev} eV",
            "edft_electrode_potential": _opt(ctx.applied_potential_v, "V"),
        }
        
@dataclass
class eDFTParams(ParamsTemplate):
    enable_attr: str | None = "enable_edft"

    enable_edft: bool = True
    enable_gc_edft: bool = True

    edft_max_it: int = 100
    edft_nelec_thres: float = 1e-6
    edft_fermi_thres: float = 1e-6
    edft_commutator_thres: float = 1e-4
    edft_smearing_width: float = 0.1  # eV

    spin_fix: int = 2
    spin: int = 0
    spin_polarised: bool = True

    gc: GCeDFTParams | None = field(default_factory=GCeDFTParams)

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        base: dict[str, Any] = {
            "edft": True,  # present iff enabled
            "edft_grand_canonical": self.enable_gc_edft,
            "edft_maxit": self.edft_max_it,
            "edft_smearing_width": _opt(self.edft_smearing_width, "eV"),
            "edft_nelec_thres": self.edft_nelec_thres,
            "edft_fermi_thres": self.edft_fermi_thres,
            "edft_commutator_thres": self.edft_commutator_thres,
            "edft_spin_fix": self.spin_fix,
            "spin_polarized": self.spin_polarised,
            "spin": self.spin,
        }
        gc_kw: dict[str, Any] = {}
        if self.enable_gc_edft and self.gc is not None:
            gc_kw = self.gc.to_keywords(ParamsTemplate.Context(applied_potential_v=ctx.applied_potential_v))
        return _merge_keywords(base, gc_kw)

@dataclass
class PBElectrolyteParams(ParamsTemplate):
    enable_attr: str | None = "enable_pbe"
    
    enable_pbe: bool = True # Corresponds to the 'is_pbe' keyword
    pbe_mode: str = "full"
    neutralisation_scheme: str = "counterions_auto"
    debye_screening: bool = True
    bcs_coarseness: int = 5
    bcs_threshold: float = 1e-7
    implicit_ions: list[dict[str, str | int | float]] = field(
        default_factory=lambda: [
            {"symbol": "H", "charge": +1, "conc": 0.1},
            {"symbol": "AuCl4", "charge": -1, "conc": 0.1},
        ]
    )

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        ions = [f"{ion['symbol']} {ion['charge']} {ion['conc']}" for ion in self.implicit_ions]
        return {
            "sol_ions": ions,
            "is_pbe": self.pbe_mode,
            "is_pbe_neutralisation_scheme": self.neutralisation_scheme,
            "is_pbe_bc_debye_screening": self.debye_screening,
            "is_pbe_temperature": _opt(ctx.temperature_k, "K"),
            "is_bc_coarseness": self.bcs_coarseness,
            "is_bc_threshold": self.bcs_threshold,
        }

@dataclass
class DLMGParams(ParamsTemplate):
    multigrid_bcs: str = "P P P"
    use_error_damping: bool = True
    use_cg: bool = True
    fd_order: int = 8
    max_res_ratio: float = 1e3
    vcyc_smoothing: int = 6
    vcycle_max_iters: int = 500
    newton_max_iters: int = 500
    steric_smearing_bohr: float = 0.4
    steric_dens_isovalue: float = 1e-3

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        v = self.vcyc_smoothing
        return {
            "multigrid_bc": self.multigrid_bcs,
            "mg_use_cg": self.use_cg,
            "mg_use_error_damping": self.use_error_damping,
            "mg_defco_fd_order": self.fd_order,
            "mg_max_res_ratio": self.max_res_ratio,
            "mg_max_iters_vcycle": self.vcycle_max_iters,
            "mg_max_iters_newton": self.newton_max_iters,
            "mg_vcyc_smoother_iter_pre": v,
            "mg_vcyc_smoother_iter_post": v,
            "is_hc_steric_smearing": f"{self.steric_smearing_bohr} Bohr",
            "is_hc_steric_dens_isovalue": self.steric_dens_isovalue,
        }

@dataclass
class SolventParams(ParamsTemplate):
    enable_attr: str | None = "enable_implicit_solvent"

    enable_implicit_solvent: bool = True # Corresponds to 'use_implicit_solvent' keyword

    dielectric_function: str = "soft_sphere"
    steric_potential_type: str = "M"
    smeared_ion_bcs: str = "P P P"
    pspot_bcs: str = "P P P"
    ion_ion_bcs: str = "P P P"

    solvent_permittivity: float = 78.4
    solvent_surf_tension_nm: str = "0.072 N/m"
    smeared_ion_width_bohr: float = 0.8

    solvent_radius_bohr: dict[str, float] = field(
        default_factory=lambda: {
            'C': 3.2, # Bohr
            'Au': 3.14 # Bohr
        })

    use_apolar_solvation: bool = True
    use_auto_solvation: bool = True
    use_smeared_ion_rep: bool = True
    use_solvation_properties: bool = True
    write_steric: bool = True

    electrolyte: PBElectrolyteParams = field(default_factory=PBElectrolyteParams)
    dlmg: DLMGParams = field(default_factory=DLMGParams)

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        radii = [f"{sym} {r:.2f}" for sym, r in self.solvent_radius_bohr.items()]
        return {
            "is_implicit_solvent": True,
            "is_auto_solvation": self.use_auto_solvation,
            "is_include_apolar": self.use_apolar_solvation,
            "is_smeared_ion_rep": self.use_smeared_ion_rep,
            "ion_ion_bc": self.ion_ion_bcs,
            "species_solvent_radius": radii,
            "smeared_ion_bc": self.smeared_ion_bcs,
            "pspot_bc": self.pspot_bcs,
            "is_dielectric_function": self.dielectric_function,
            "is_steric_pot_type": self.steric_potential_type,
            "is_bulk_permittivity": self.solvent_permittivity,
            "is_solvent_surf_tension": self.solvent_surf_tension_nm,
            "is_solvation_properties": self.use_solvation_properties,
            "is_steric_write": self.write_steric,
            "is_smeared_ion_width": self.smeared_ion_width_bohr,
            "is_pbe_energy_tolerance" : "1E+50 Ha",
        }

@dataclass
class SlurmParams:
    """SLURM parameters; used only for writing the sbatch script from a template."""
    partition: str = "compute"
    nodes: int = 1
    tasks_per_node: int = 3
    cpus_per_task: int = 7
    walltime: str = "48:00:00"
    mem_per_cpu_mb: int = 4500
    job_name: str = ""
    onetep_binary: str = "/springbrook/share/physicsnh/onetep_main/bin/onetep.blythe_gnu"
    onetep_launcher: str = "/springbrook/share/physicsnh/onetep_main/utils/onetep_launcher"
    template_path: Path = Path("templates/sbatch_template.sh")

@dataclass
class OnetepParams:
    core: CoreParams = field(default_factory=CoreParams)
    io: OnetepIO = field(default_factory=OnetepIO)
    ngwfs: NGWFParams = field(default_factory=NGWFParams)
    edft: eDFTParams = field(default_factory=eDFTParams)
    solvent: SolventParams = field(default_factory=SolventParams)
    geom: GeometryParams = field(default_factory=GeometryParams)
    slurm: SlurmParams = field(default_factory=SlurmParams)

    extra: dict[str, Any] = field(default_factory=dict)

    def build_profile(self, mpi_ranks: int) -> OnetepProfile:
        """
        A fallback command so ASE can write inputs locally.
        Production runs should use the sbatch script that calls srun via your launcher.
        """
        cmd = (
            f"mpirun -np {mpi_ranks} {self.slurm.onetep_binary} {{seed}}.dat "
            f"> {{seed}}.out 2> {{seed}}.err"
        )
        return OnetepProfile(command=cmd, pseudo_path=self.core.pseudo_path)
    
    def iter_applied_potentials(self) -> list[float | None]:
        if (
            self.edft.enable_edft
            and self.edft.enable_gc_edft
            and self.edft.gc is not None
            and len(self.edft.gc.electrode_potentials_v) > 0
        ):
            return list(self.edft.gc.electrode_potentials_v)
        return [None]

    def build_keywords(self, applied_potential_v: float | None) -> KeywordDict:
        """
        Compose the final ONETEP keyword dictionary.
        Each pack contributes only when its internal enable flags allow it.
        """
        ctx = ParamsTemplate.Context(
            temperature_k=self.core.temperature_k,
            applied_potential_v=applied_potential_v,
        )

        parts: list[KeywordDict] = []

        # Include core, io, ngwfs classes
        parts.append(self.core.to_keywords(ctx))
        parts.append(self.io.to_keywords(ctx))
        parts.append(self.ngwfs.to_keywords(ctx))

        # Include eDFT and GC-eDFT if enabled
        parts.append(self.edft.to_keywords(ctx))

        # Include solvent if enabled
        solvent_kw = self.solvent.to_keywords(ctx)
        parts.append(solvent_kw)

        # If solvent is enabled, PB + DLMG may contribute
        if solvent_kw:
            parts.append(self.solvent.electrolyte.to_keywords(ctx))
            parts.append(self.solvent.dlmg.to_keywords(ctx))
            
        # If using GeometryOptimization and geometry constraints are used, include them
        if self.core.task == "GeometryOptimization":
            parts.append({"species_constraints": self.core.geom_constraints.species_constraints()} if self.core.geom_constraints.use_constraints else {})
            
        # Final ad-hoc overrides
        parts.append(self.extra)

        return _merge_keywords(*parts)
    
