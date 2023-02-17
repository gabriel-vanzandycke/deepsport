#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres="gpu:1"
#SBATCH --partition=gpu

#workers=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`

REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --file)
            cp "$GLOBALSCRATCH/$2" "$LOCALSCRATCH/$2"
            shift # past argument
            shift # past value
            ;;
        *)
            REMAINING_ARGS+=($1) # save positional arg
            shift # past argument
            ;;
    esac
done

python -m experimentator ${REMAINING_ARGS[@]} --kwargs "jobid=${SLURM_JOB_ID}"
