#!/bin/bash
#SBATCH -p normal
#SBATCH -N 32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH -J uw-gpu
#SBATCH -o %j.out
#SBATCH -e %j.err

module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/intel-2017.5.239
module load compiler/rocm/dtk/22.04.2
export OMP_NUM_THREADS=2
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
rm `pwd`/hostfile-dl -f

for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)

np=$(($np*4))

nodename=$(cat $hostfile |sed -n "1p")
echo $nodename
dist_url=`echo $nodename | awk '{print $1}'`


echo mpirun -np $np --allow-run-as-root --hostfile hostfile-dl-$SLURM_JOB_ID  --bind-to none ./uwshcu.exe $dist_url resnet50 64
mpirun -np $np --allow-run-as-root --hostfile hostfile-dl-$SLURM_JOB_ID --bind-to none rocm-profiler -C -O --counterfile /opt/rocm/profiler/counterfiles/counters_HSA_gfx906_pass1.csl --counterfile /opt/rocm/profiler/counterfiles/counters_OpenCL_gfx906_pass2.csl ./uwshcu.exe $dist_url resnet50 64
