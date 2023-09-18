#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --gres="gpu:1"
#SBATCH --partition=gpu
#SBATCH --exclude=mb-mil102,mb-rom101,mb-rom102,mb-rom103

#workers=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
workers=0

REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --mem)
            workers=$((`nvidia-smi --query-gpu=index,memory.total --format=csv | tail -1 | awk '{print $2}'` / $2))
            echo "With memory limit of $2, $workers workers will be launched"
            shift # past argument
            shift # past value
            ;;
        *)
            REMAINING_ARGS+=($1) # save positional arg
            shift # past argument
            ;;
    esac
done

python -m experimentator ${REMAINING_ARGS[@]} --kwargs "jobid=${SLURM_JOB_ID}" --workers $workers
