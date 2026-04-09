#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A LASSCF_gpudev
#PBS -l filesystems=home:grand
#PBS -N fci_miniapps
#export NV_ACC_DEBUG=1
#PBS -m be
#PBS -M valayagarawal@uchicago.edu
cd /lus/grand/projects/LASSCF_gpudev/valayagarawal/soft/original/mrh2/mrh/gpu/mini-apps/fci/loop_tdm

. /lus/grand/projects/LASSCF_gpudev/valayagarawal/scripts/setup_polaris_3.sh

# MPI example w/ 16 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NTHREADS=32
NDEPTH=${NTHREADS}

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

#MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} "
MPI_ARGS="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth "

OMP_ARGS=" "
#OMP_ARGS="--env OMP_NUM_THREADS=${NTHREADS} "
OMP_ARGS=" --env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=threads "
#OMP_ARGS+=" --env OMP_WAIT_POLICY=ACTIVE "

#INPUT="dump/inp1/4_sto-3g_gpu_inp.py"
#INPUT="1_6-31g_inp.py"
#INPUT="1_6-31g_inp_gpu.py"
#INPUT="1_6-31g_inp_scf_gpu.py"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYSCF_MAX_MEMORY=160000
#INPUT="inp_88.py"
#EXE="python ${INPUT} "
#{ time mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} ;} 2>&1 | tee gpu_profile.txt
#INPUT="inp_66.py"
#EXE="python ${INPUT} "
#{ time mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} ;} 2>&1 | tee gpu_profile.txt
INPUT="tdm13h_performance.py"
EXE="python ${INPUT} "
#{ time mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} ;} 2>&1 | tee gpu_profile.txt
nsys profile --stats=true -t cuda,nvtx mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} 2>&1 | tee profile.txt

#nsys profile --stats=true -t cuda,nvtx mpiexec ${MPI_ARGS} ${OMP_ARGS} ${EXE} 2>&1 | tee profile.txt

#EXE=/home/knight/repos/GettingStarted/Examples/Polaris/affinity_omp/hello_affinity
#EXE="python my_profile.py ${INPUT} "

#mpiexec ${MPI_ARGS} ${OMP_ARGS} /home/knight/repos/GettingStarted/Examples/Polaris/affinity_omp/hello_affinity 

#python -m cProfile -o out.prof ${INPUT}
#{ time ${EXE} ;} 2>&1 | tee profile.txt
#cuda-gdb --args ${EXE}

#export OMP_NUM_THREADS=$NTHREADS
#ncu -s 3 -c 100 --print-summary per-kernel ${EXE} 2>&1 | tee profile.txt
