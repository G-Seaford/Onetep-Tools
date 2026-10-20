#!/usr/bin/env python3
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from collections import ChainMap

from typing import Any, Literal, Optional
from ase.calculators.onetep import OnetepProfile

# Typing aliases
KeywordValue = str | int | float | bool | list[str]
KeywordDict = dict[str, KeywordValue]

# Allowed systems
System = Literal["Blythe", "Sulis", "Archer2", "Isambard-AI", "default"]

# Sentinel for unset values
class _UnsetType:
    """Marks a parameter as not set: its keyword is omitted from the .dat."""
    _instance: _UnsetType | None = None

    def __new__(cls):
        if cls._instance is None: cls._instance = super().__new__(cls)
        return cls._instance

    # Singleton survives copy/deepcopy/pickle, so `is UNSET` always holds
    def __repr__(self) -> str: return "UNSET"
    def __bool__(self) -> bool: return False
    def __copy__(self): return self
    def __deepcopy__(self, memo): return self
    def __reduce__(self): return (_UnsetType, ())

UNSET = _UnsetType()

def is_set(v: Any) -> bool:
    return v is not UNSET

def or_unset(v: Any) -> Any:
    """Normalise None (argparse, call sites) to UNSET at the boundary."""
    return UNSET if v is None else v

# Helpers
def _strip_unset(d: dict[str, Any]) -> KeywordDict:
    leftover = [k for k, v in d.items() if v is None]
    if leftover: raise ValueError(f"Keywords set to None (use UNSET): {leftover}")
    return {k: v for k, v in d.items() if is_set(v)}

def _merge_keywords(*dicts: dict[str, Any]) -> KeywordDict:
    dicts = [d for d in dicts if d]
    merged = dict(ChainMap(*reversed(list(dicts))))
    return _strip_unset(merged)

# Physical units accepted by ONETEP's input parser, by dimension (phy table in
# src/esdf_mod.F90). ONETEP matches unit names case-insensitively.
ONETEP_UNITS: dict[str, frozenset[str]] = {
    "energy":          frozenset({"ha", "hartree", "mha", "mhartree", "ev", "mev", "ry", "mry", "j", "erg",
                                  "kcal/mol", "kj/mol", "hz", "thz", "cm-1", "cm^-1", "cm**-1", "k"}),
    "length":          frozenset({"bohr", "ang", "nm", "cm", "m"}),
    "inverse_length":  frozenset({"1/bohr", "bohr-1", "bohr^-1", "bohr**-1", "1/ang", "ang-1", "ang^-1",
                                  "ang**-1", "1/nm", "nm-1", "nm^-1", "nm**-1"}),
    "force":           frozenset({"ha/bohr", "ry/bohr", "ev/ang", "n"}),
    "pressure":        frozenset({"ha/bohr**3", "ry/bohr**3", "ev/ang**3", "pa", "mpa", "gpa", "atm", "bar", "mbar"}),
    "surface_tension": frozenset({"ha/bohr**2", "j/m**2", "n/m"}),
    "potential":       frozenset({"v", "ha/e"}),
    "efield":          frozenset({"ha/bohr/e", "ry/bohr/e", "v/bohr", "v/ang", "v/nm", "v/m"}),
    "time":            frozenset({"aut", "fs", "ps", "ns", "s"}),
}

# A physical value: bare number, (value, unit) or "value unit", e.g. 800, (30, "Ha"), "30 Ha"
Quantity = float | tuple[float, str] | str

def _with_unit(v: Any, dim: str, default_unit: str | None = None) -> Any:
    """
    Format a physical keyword value. A bare number gets `default_unit` (or no
    unit, so ONETEP's own default applies). An explicit unit, given as
    (value, unit) or "value unit", must be one ONETEP accepts for `dim`.
    """
    if not is_set(v): return UNSET
    if isinstance(v, str):
        parts = v.split()
        if len(parts) == 1: v = parts[0]
        elif len(parts) == 2: v = tuple(parts)
        else: raise ValueError(f"Cannot parse physical value {v!r}; expected 'value unit'")
    if isinstance(v, (tuple, list)):
        if len(v) != 2: raise ValueError(f"Physical value {v!r} must be (value, unit)")
        value, unit = v
        float(value)  # raises if the value is not numeric
        if unit.lower() not in ONETEP_UNITS[dim]:
            raise ValueError(f"Unit {unit!r} is not a valid ONETEP {dim} unit; "
                             f"allowed (case-insensitive): {', '.join(sorted(ONETEP_UNITS[dim]))}")
        return f"{value} {unit}"
    return f"{v} {default_unit}" if default_unit else v

def _vec(v: Any) -> Any:
    return " ".join(map(str, v)) if is_set(v) else UNSET

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
    apply_vacuum: bool = False
    vacuum: float = 0.0
    vacuum_axes: tuple[int, ...] = (2,)
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class OnetepIO(ParamsTemplate):
    """ONETEP I/O and formatting (no write_forces here by design)."""
    output_detail: str = "normal"
    timings_level: int | _UnsetType = UNSET
    
    # Output formats
    dx_format: bool | _UnsetType = UNSET
    cube_format: bool | _UnsetType = UNSET

    # Write options
    write_xyz: bool = False
    write_tightbox_ngwfs: bool = False
    write_hamiltonian: bool = False
    write_denskern: bool = False
    
    # Read options
    read_tightbox_ngwfs: bool = False
    read_hamiltonian: bool = False
    read_denskern: bool = False

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        return {
            "output_detail": self.output_detail,
            "forces_output_detail": self.output_detail,
            "timings_level": self.timings_level,
            "dx_format": self.dx_format,
            "cube_format": self.cube_format,
            "write_xyz": self.write_xyz,
            "write_tightbox_ngwfs": self.write_tightbox_ngwfs,
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
    xc_functional: str | _UnsetType = UNSET
    kpoint_method: str | _UnsetType = UNSET
    use_paw: bool = True
    paw_core_den_taper_gvec_frac: float | _UnsetType = UNSET
    write_forces: bool = True
    write_properties: bool = False

    # Paths & IO
    pseudo_path: str =""
    pseudopotentials: dict[str, str] =field(default_factory=dict)

    input_path: Path = Path(".")
    output_path: Path =Path(".")
    seed: str | _UnsetType = UNSET

    # System parameters
    cutoff_energy: Quantity | _UnsetType = UNSET  # bare number: eV
    kernel_cutoff: Quantity | _UnsetType = UNSET  # bare number: bohr
    kpoint_grid: tuple[int, int, int] | _UnsetType = UNSET
    kpar_groups: int | _UnsetType = UNSET
    temperature_k: float | _UnsetType = UNSET
    fine_grid_scale: float | _UnsetType = UNSET

    # Fast algorithms
    fast_density: bool | _UnsetType = UNSET
    fast_locpot_int: bool | _UnsetType = UNSET
    fast_ngwf_gradient: bool | _UnsetType = UNSET

    trimmed_boxes_threshold: float | _UnsetType = UNSET
    threads_num_fftboxes: int | _UnsetType = UNSET
    threads_gpu : int | _UnsetType = UNSET
    comms_group_size: int | _UnsetType = UNSET
    
    # Symmetry Operations
    use_symmetry: bool | _UnsetType = UNSET
    use_time_reversal: bool | _UnsetType = UNSET
    
    # Geometry constraints
    geom_constraints: GeomConstraints = field(default_factory=GeomConstraints)

    def _paw_core_taper(self) -> float | _UnsetType:
        """PAW pseudo-core taper, passed only when fine_grid_scale > 2.0 (ONETEP default 2.0)."""
        if not is_set(self.paw_core_den_taper_gvec_frac): return UNSET
        scale = self.fine_grid_scale if is_set(self.fine_grid_scale) else 2.0
        if scale <= 2.0:
            logging.warning("Ignoring paw_core_den_taper_gvec_frac=%s: requires fine_grid_scale > 2.0 (got %s).",
                            self.paw_core_den_taper_gvec_frac, scale)
            return UNSET
        return self.paw_core_den_taper_gvec_frac

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        return {
            "task": self.task,
            "cutoff_energy": _with_unit(self.cutoff_energy, "energy", "eV"),
            "kernel_cutoff": _with_unit(self.kernel_cutoff, "length", "bohr"),
            "xc_functional": self.xc_functional,
            "paw": self.use_paw,
            "paw_core_den_taper_gvec_frac": self.paw_core_den_taper_gvec_frac,
            "write_forces": self.write_forces,
            'do_properties' : self.write_properties,
            "kpoint_method": self.kpoint_method,
            "kpoint_grid_size": _vec(self.kpoint_grid),
            "num_kpars": self.kpar_groups,
            "use_symmetry": self.use_symmetry,
            "use_time_reversal": self.use_time_reversal,
            "fine_grid_scale": self.fine_grid_scale,
            "fast_density": self.fast_density,
            "fast_locpot_int": self.fast_locpot_int,
            "fast_ngwf_gradient": self.fast_ngwf_gradient,
            "trimmed_boxes_threshold": self.trimmed_boxes_threshold,
            "threads_num_fftboxes": self.threads_num_fftboxes,
            "threads_gpu": self.threads_gpu,
            "comms_group_size": self.comms_group_size,
        }

@dataclass
class NGWFParams(ParamsTemplate):
    """NGWF settings."""
    extend_ngwfs: tuple[bool, bool, bool] | _UnsetType = UNSET
    ngwfs_count: dict[str, int] | _UnsetType = UNSET
    ngwfs_radius: dict[str, float] | _UnsetType = UNSET

    maxit_ngwf_cg: int | _UnsetType = UNSET
    ngwf_threshold_orig: float | _UnsetType = UNSET

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        return {
            "extend_ngwf": _vec(self.extend_ngwfs),
            "maxit_ngwf_cg": self.maxit_ngwf_cg,
            "ngwf_threshold_orig": self.ngwf_threshold_orig,
        }

@dataclass
class GCeDFTParams(ParamsTemplate):
    
    reference_potential_ev: Quantity | _UnsetType = UNSET  # bare number: eV
    electrode_potentials_v: list[float] | _UnsetType = UNSET

    @property
    def electrode_potentials(self) -> list[float]:
        """Read-only-ish view (copy) of the sweep as a list."""
        return list(self.electrode_potentials_v) if is_set(self.electrode_potentials_v) else []

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        return {
            "edft_reference_potential": _with_unit(self.reference_potential_ev, "energy", "eV"),
            "edft_electrode_potential": _with_unit(ctx.applied_potential_v, "potential", "V") if is_set(self.electrode_potentials_v) else UNSET,
        }
        
@dataclass
class eDFTParams(ParamsTemplate):
    enable_attr: str | None = "enable_edft"

    enable_edft: bool = False
    enable_gc_edft: bool = False

    edft_max_it: int | _UnsetType = UNSET
    edft_nelec_thres: float | _UnsetType = UNSET
    edft_fermi_thres: float | _UnsetType = UNSET
    edft_commutator_thres: float | _UnsetType = UNSET
    edft_smearing_width: Quantity | _UnsetType = UNSET  # bare number: eV

    spin_fix: int | _UnsetType = UNSET
    spin: int | _UnsetType = UNSET
    spin_polarised: bool | _UnsetType = UNSET

    gc: GCeDFTParams = field(default_factory=GCeDFTParams)

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        base: dict[str, Any] = {
            "edft": True,  # present iff enabled
            "edft_grand_canonical": self.enable_gc_edft,
            "edft_maxit": self.edft_max_it,
            "edft_smearing_width": _with_unit(self.edft_smearing_width, "energy", "eV"),
            "edft_nelec_thres": self.edft_nelec_thres,
            "edft_fermi_thres": self.edft_fermi_thres,
            "edft_commutator_thres": self.edft_commutator_thres,
            "edft_spin_fix": self.spin_fix,
            "spin_polarized": self.spin_polarised,
            "spin": self.spin,
        }
        gc_kw = self.gc.to_keywords(ParamsTemplate.Context(applied_potential_v=ctx.applied_potential_v)) if self.enable_gc_edft else {}
        return _merge_keywords(base, gc_kw)

@dataclass
class PBElectrolyteParams(ParamsTemplate):
    enable_attr: str | None = "enable_pbe"
    
    enable_pbe: bool = False # Corresponds to the 'is_pbe' keyword
    pbe_mode: str | _UnsetType = UNSET
    neutralisation_scheme: str | _UnsetType = UNSET
    debye_screening: bool | _UnsetType = UNSET
    bcs_coarseness: int | _UnsetType = UNSET
    bcs_threshold: float | _UnsetType = UNSET
    implicit_ions: list[dict[str, str | int | float]] | _UnsetType = UNSET

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        ions = [f"{ion['symbol']} {ion['charge']} {ion['conc']}" for ion in self.implicit_ions] if is_set(self.implicit_ions) else UNSET
        return {
            "sol_ions": ions,
            "is_pbe": self.pbe_mode,
            "is_pbe_neutralisation_scheme": self.neutralisation_scheme,
            "is_pbe_bc_debye_screening": self.debye_screening,
            # Plain number in ONETEP (always kelvin, not a physical keyword), so no unit choice
            "is_pbe_temperature": f"{ctx.temperature_k} K" if is_set(ctx.temperature_k) else UNSET,
            "is_bc_coarseness": self.bcs_coarseness,
            "is_bc_threshold": self.bcs_threshold,
        }

@dataclass
class DLMGParams(ParamsTemplate):
    multigrid_bcs: str | _UnsetType = UNSET
    use_error_damping: bool | _UnsetType = UNSET
    use_cg: bool | _UnsetType = UNSET
    fd_order: int | _UnsetType = UNSET
    max_res_ratio: float | _UnsetType = UNSET
    vcyc_smoothing: int | _UnsetType = UNSET
    vcycle_max_iters: int | _UnsetType = UNSET
    newton_max_iters: int | _UnsetType = UNSET
    steric_smearing_bohr: Quantity | _UnsetType = UNSET  # bare number: bohr
    steric_dens_isovalue: float | _UnsetType = UNSET

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
            "is_hc_steric_smearing": _with_unit(self.steric_smearing_bohr, "length", "Bohr"),
            "is_hc_steric_dens_isovalue": self.steric_dens_isovalue,
        }

@dataclass
class SolventParams(ParamsTemplate):
    enable_attr: str | None = "enable_implicit_solvent"

    enable_implicit_solvent: bool = False # Corresponds to 'use_implicit_solvent' keyword

    dielectric_function: str | _UnsetType = UNSET
    steric_potential_type: str | _UnsetType = UNSET
    smeared_ion_bcs: str | _UnsetType = UNSET
    pspot_bcs: str | _UnsetType = UNSET
    ion_ion_bcs: str | _UnsetType = UNSET

    solvent_permittivity: float | _UnsetType = UNSET
    solvent_surf_tension_nm: Quantity | _UnsetType = UNSET  # bare number: ONETEP default (Ha/bohr**2)
    smeared_ion_width_bohr: Quantity | _UnsetType = UNSET   # bare number: ONETEP default (bohr)

    solvent_radius_bohr: dict[str, float] | _UnsetType = UNSET

    use_apolar_solvation: bool | _UnsetType = UNSET
    use_auto_solvation: bool | _UnsetType = UNSET
    use_smeared_ion_rep: bool | _UnsetType = UNSET
    use_solvation_properties: bool | _UnsetType = UNSET
    write_steric: bool | _UnsetType = UNSET

    electrolyte: PBElectrolyteParams = field(default_factory=PBElectrolyteParams)
    dlmg: DLMGParams = field(default_factory=DLMGParams)

    def _core_keywords(self, ctx: ParamsTemplate.Context) -> dict[str, Any]:
        radii = [f"{sym} {r:.2f}" for sym, r in self.solvent_radius_bohr.items()] if is_set(self.solvent_radius_bohr) else UNSET
        return {
            "is_implicit_solvent": self.enable_implicit_solvent,
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
            "is_solvent_surf_tension": _with_unit(self.solvent_surf_tension_nm, "surface_tension"),
            "is_solvation_properties": self.use_solvation_properties,
            "is_steric_write": self.write_steric,
            "is_smeared_ion_width": _with_unit(self.smeared_ion_width_bohr, "length"),
            "is_pbe_energy_tolerance" : "1E+50 Ha",
        }

@dataclass
class SlurmParams:
    system: System = "default"
    partition: str | _UnsetType = UNSET
    nodes: int | _UnsetType = UNSET
    gpus: int | _UnsetType = UNSET

    tasks_per_node: int | _UnsetType = UNSET
    cpus_per_task: int | _UnsetType = UNSET
    walltime: str | _UnsetType = UNSET
    mem_per_cpu_mb: int | _UnsetType = UNSET
    job_name: str | _UnsetType = UNSET

    # Optional account and QoS
    account: str | _UnsetType = UNSET
    qos: str | _UnsetType = UNSET

    # ONETEP paths
    onetep_binary: str | _UnsetType = UNSET
    onetep_launcher: str | _UnsetType = UNSET


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
    
    def iter_applied_potentials(self) -> list[float | _UnsetType]:
        if self.edft.enable_edft and self.edft.enable_gc_edft and self.edft.gc.electrode_potentials:
            return self.edft.gc.electrode_potentials
        return [UNSET]

    def build_keywords(self, applied_potential_v: float | _UnsetType | None = UNSET) -> KeywordDict:
        """
        Compose the final ONETEP keyword dictionary.
        Each pack contributes only when its internal enable flags allow it.
        """
        ctx = ParamsTemplate.Context(
            temperature_k=self.core.temperature_k,
            applied_potential_v=or_unset(applied_potential_v),
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
        if self.solvent.enable_implicit_solvent:
            parts.append(self.solvent.electrolyte.to_keywords(ctx))
            parts.append(self.solvent.dlmg.to_keywords(ctx))
            
        # If using GeometryOptimization and geometry constraints are used, include them
        if self.core.task == "GeometryOptimization":
            parts.append({"species_constraints": self.core.geom_constraints.species_constraints()} if self.core.geom_constraints.use_constraints else {})
            
        # Final ad-hoc overrides
        parts.append(self.extra)

        return _merge_keywords(*parts)
    
