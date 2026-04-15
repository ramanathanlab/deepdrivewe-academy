# DeepDriveWE-Academy

**Weighted ensemble molecular dynamics using the Academy multi-agent framework.**

DeepDriveWE-Academy is a Python implementation of
[DeepDriveWE](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c01136)
built on the [Academy](https://docs.academy-agents.org/stable/) multi-agent
framework. It enables scalable weighted ensemble (WE) simulations where
molecular dynamics walkers and resampling logic run as independent,
communicating agents distributed across CPUs and GPUs.

## Key Features

- **Agent-based architecture** -- Simulation walkers and the WE
  orchestrator are Academy agents that communicate asynchronously via
  typed messages (`SimResult`, `SimMetadata`).
- **Pluggable components** -- Swap binners, recyclers, resamplers, and
  simulation engines independently.
- **GPU-distributed MD** -- Simulations are offloaded to GPU workers
  through [Parsl](https://parsl-project.org/), with built-in support for
  workstations, Slurm, and PBS clusters.
- **Checkpointing and resume** -- Ensemble state is saved after every
  iteration in both JSON and WESTPA-compatible HDF5 format.
- **AI-enhanced resampling** -- Optional deep learning models
  (convolutional VAE, adversarial autoencoders) for latent-space--guided
  resampling.

## Quick Example

```python
from deepdrivewe.workflows import run_westpa_workflow

await run_westpa_workflow(
    manager=manager,
    sim_agent_type=MySimAgent,
    westpa_agent_type=MyWestpaAgent,
    max_iterations=100,
    ensemble=ensemble,
)
```

See the [Quickstart](getting-started/quickstart.md) guide to get running
in minutes, or follow the full
[OpenMM NTL9 tutorial](tutorials/openmm-ntl9-hk.md) for a production
protein-folding workflow.

!!! tip "Build with Claude Code"
    This project ships a [Claude Code](https://claude.com/claude-code)
    skill that walks Claude through creating a new DeepDriveWE example
    end-to-end. See the
    [Claude Skill](getting-started/claude-skill.md) page to install
    it, then ask Claude things like *"build a new DeepDriveWE example
    for ligand unbinding with AMBER"*.

## Citation

If you use DeepDriveWE in your research, please cite:

> Leung, J. M. G.; Frazee, N. C.; Brace, A.; Bogetti, A. T.;
> Ramanathan, A.; Chong, L. T. "Unsupervised Learning of Progress
> Coordinates during Weighted Ensemble Simulations: Application to NTL9
> Protein Folding." *Journal of Chemical Theory and Computation* **2025**,
> *21* (7), 3691--3699.
> [DOI: 10.1021/acs.jctc.4c01136](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c01136)

BibTeX:

```bibtex
@article{leung2025unsupervised,
  title={Unsupervised Learning of Progress Coordinates during Weighted Ensemble Simulations: Application to NTL9 Protein Folding},
  author={Leung, Jeremy MG and Frazee, Nicolas C and Brace, Alexander and Bogetti, Anthony T and Ramanathan, Arvind and Chong, Lillian T},
  journal={Journal of chemical theory and computation},
  volume={21},
  number={7},
  pages={3691--3699},
  year={2025},
  publisher={ACS Publications}
}
```

## License

DeepDriveWE-Academy is released under the
[MIT License](https://github.com/ramanathanlab/deepdrivewe-academy/blob/main/LICENSE).
