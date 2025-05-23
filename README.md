# Scalable Climate Modeling with Mixed-Precision Computing and Load Balancing Methods over Large-Scale Heterogeneous Computing Systems

## Project Overview

This project aims to enhance the performance of climate models on large-scale heterogeneous computing systems through mixed-precision computing and load balancing techniques. It includes the original serial Fortran code for the shallow cumulus convection physical process, as well as optimized versions for heterogeneous computing (including GPU acceleration, mixed-precision computing, and load balancing). These optimized versions significantly improve computational efficiency. On the ORISE supercomputing system, the combination of mixed-precision and load balancing algorithms achieves a floating-point performance of 1.9565 PFLOP/s on 4096 nodes.

The main contributions of the project are:

- **Mixed-Precision Algorithm**: Reduces computational cost by performing mixed single- and double-precision parallel computing on GPUs.
- **Load Balancing Method**: Optimizes resource utilization by balancing computational tasks between CPUs and GPUs using two optional task allocation strategies.
- **Error Compensation Scheme**: Enhances simulation accuracy by compensating for errors from low-precision GPU computations with double-precision CPU computations.

The project code and related resources are archived on Zenodo with the DOI: `10.5281/zenodo.15266176`.

## Installation Guide


### Installation Steps


1. **Compile the Code**:

Navigate to the respective subdirectory under `src/` and compile using `hipcc`.

For example, to compile the mixed-precision UWshcu code:

```bash
cd src/mixed-precisionUWshcu
hipcc -o mixed_uwshcu main.cu
```

2. **Integration with CAS-ESM IAP**:

Use the `Makefile` and `Macros.cnic_419` in the `scripts/` directory for compilation and execution.

## Usage Instructions

### Running Scripts

- **Double-Precision and Mixed-Precision UWshcu**:

Use the `scripts/mpihip.sh` script to run.

Example command:

```bash
sbatch scripts/mpihip.sh
```

- **Double-Precision and Mixed-Precision Load Balancing**:

Use the `scripts/loadbalance.sh` script to run.

Example command:
```bash
sbatch scripts/loadbalance.sh
```

