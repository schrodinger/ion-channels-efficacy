# Ligand Efficacy Modeling on Ion Channels

Supporting materials for our project on ligand efficacy across ion channels: analysis code, machine-readable results, and FEP starting poses. 

The analysis code in the folder analysis reads the results and labels and produces the corresponding plots. For each target, we provide the FEP starting poses as Maestro files (.mae). Restraints can be applied to FEP+ via the `-restraints-file` flag (harmonic only) or the custom code maintained [here](https://github.com/schrodinger/fep-restraints).

We have investigated the following targets:

- GluA2: glutamate ionotropic receptor AMPA type subunit 2
- GABAAR ρ1: gamma-aminobutyric acid type A receptor ρ1 subunit
- α3β4 nAChR: nicotinic acetylcholine receptor subtype α3β4
- 5-HT3AR: serotonin 3A receptor
- TRPML1: mucolipin TRP cation channel 1
- KCNQ2: potassium voltage-gated channel subfamily Q member 2


