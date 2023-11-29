#!/bin/bash
#SBATCH --job-name=uw-gpu
#SBATCH --partition=normal
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4
#SBATCH --exclusive
##SBATCH -n 32

module load compiler/devtoolset/7.3.1
#module load mpi/hpcx/2.4.1/gcc-7.3.1
module load mpi/hpcx/2.7.4/intel-2017.5.239
#module load mpi/intelmpi/2021.1.1

ulimit -s unlimited
ulimit -v unlimited
ulimit -d unlimited
#mpirun -x LD_PRELOAD=mpitrace.so:/opt/hpc/software/mpi/hpcx/v2.7.4/gcc-7.3.1/lib/libmpi.so.40 -np 64 hipprof --mpi-trace ./uwshcu.exe
mpirun -np 4 ./uwshcu.exe

#srun hostname |sort| uniq -c |awk ‘{printf“%s:%s\n”,$2,$1}’ > hostfile
#mpirun -np ${test_np} --hostfile ${hostfile}\
#       --mca plm_rsh_no_tree_spawn 1 \
#	   --mca plm_rsh_num_concurrent ${node_num} \
#	   -mca routed_radix ${node_num} \
#	   -mca pml ucx \
#	   -x LD_LIBRARY_PATH \
#	   --bind-to none`pwd`/run.sh
