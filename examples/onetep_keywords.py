DEFAULT_PARAMETERS = {
    "core": {
        "task": "SinglePoint",
        "xc_functional": "OPTB88",
        "kpoint_method": "PW",
        "use_paw": True,
        "write_forces": True,

        "pseudo_path": "/springbrook/home/p/phumhp/Au-Graphene/pseudos",
        "pseudopotentials": {"C": "C.PBE-paw.abinit", "Au": "Au.PBE-paw.abinit"},

        "output_path": "Calculations/",
        "cutoff_energy_ev": 750,
        "kpoint_grid": (3, 3, 1),
        "kpar_groups": 1,
        "temperature_k": 298.15,
        
        "use_symmetry": True,
        "use_time_reversal": True,
        "fine_grid_scale": 4.0,
    },

    "io": {
        "output_detail": "verbose",
        "dx_format": False,
        "cube_format": True,
        "write_xyz": True,
        "write_hamiltonian": True,
        "write_denskern": True,
        "read_tightbox_ngwfs": False,
        "read_hamiltonian": False,
        "read_denskern": False,
    },

    "ngwfs": {
        "extend_ngwfs": (True, True, True),
        "write_tightbox_ngwfs": True,
        "ngwfs_count": {"C": -1, "Au": -1},
        "ngwfs_radius": {"C": 10.0, "Au": 12.0},
    },

    "edft": {
        "enable_edft": True,
        "enable_gc_edft": True, # Remember to enable/disable electrolyte lol
        "edft_max_it": 35,
        "edft_nelec_thres": 1e-6,
        "edft_fermi_thres": 1e-6,
        "edft_commutator_thres": 1e-4,
        "spin_fix": 2,
        "spin": 0,
        "spin_polarised": False,
        
        "gc": {
            "reference_potential_ev": -2.70,
            "electrode_potentials_v": [0.00,],
        },
    },

    "solvent": {
        "enable_implicit_solvent": True,
        "dielectric_function": "soft_sphere",
        "steric_potential_type": "M",
        "smeared_ion_bcs": "P P P",
        "pspot_bcs": "P P P",
        "ion_ion_bcs": "P P P",
        "solvent_permittivity": 78.4,
        "solvent_surf_tension_nm": "0.072 N/m",
        "smeared_ion_width_bohr": 0.8,
        "solvent_radius_bohr": {"C": 3.20,
                                "Au": 3.14},
        "use_apolar_solvation": True,
        "use_auto_solvation": True,
        "use_smeared_ion_rep": True,
        "use_solvation_properties": True,
        "write_steric": False,
        
        "electrolyte": {
            "enable_pbe": True,
            "pbe_mode": "full",
            "neutralisation_scheme": "counterions_auto",
            "debye_screening": True,
            "bcs_coarseness": 5,
            "bcs_threshold": 1e-7,
            "implicit_ions": [
                {"symbol": "H", "charge": +1, "conc": 0.1},
                {"symbol": "AuCl4", "charge": -1, "conc": 0.1},
            ],
        },
        
        "dlmg": {
            "multigrid_bcs": "P P P",
            "use_error_damping": True,
            "use_cg": True,
            "fd_order": 8,
            "max_res_ratio": 1e3,
            "vcyc_smoothing": 6,
            "vcycle_max_iters": 500,
            "newton_max_iters": 500,
            "steric_smearing_bohr": 0.4,
            "steric_dens_isovalue": 1e-3,
        },
    },

    "slurm": {
        "partition": "compute",
        "nodes": 1,
        "tasks_per_node": 12,
        "cpus_per_task": 7,
        "walltime": "48:00:00",
        "mem_per_cpu_mb": 4500,
        "job_name": "",
        "onetep_binary": "/springbrook/share/physicsnh/onetep_main/bin/onetep.blythe_gnu",
        "onetep_launcher": "/springbrook/share/physicsnh/onetep_main/utils/onetep_launcher",
    },
}
