#!/bin/bash
#SBATCH --job-name=rlvr_training
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rlvr_training_%j.out
#SBATCH --error=logs/rlvr_training_%j.err

# ============================================================================
# Distributed RLVR Training SLURM Script
# 
# Configuration: 20 nodes total
#   - 16 training nodes (64 GPUs) - ONE distributed training job
#   - 4 vLLM inference nodes (16 GPUs) - load balanced
#
# Node Layout:
#   Training: nodes 0-3, 5-8, 10-13, 15-18 (16 nodes)
#   vLLM:     nodes 4, 9, 14, 19 (4 nodes)
#
# Each training node group connects to its nearest vLLM server:
#   nodes 0-3   -> vLLM on node 4
#   nodes 5-8   -> vLLM on node 9
#   nodes 10-13 -> vLLM on node 14
#   nodes 15-18 -> vLLM on node 19
# ============================================================================

# Create log directory
mkdir -p logs

# Load required modules (adjust based on your cluster)
module load cuda/12.1
module load python/3.10

# Activate virtual environment
source ~/venv/bin/activate

# Set environment variables
export WANDB_PROJECT="mini_rlvr_test"
export WANDB_ENTITY="twu376@wisc.edu"
export HF_TOKEN="${HF_TOKEN:-}"
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo "Allocated nodes: ${NODELIST[@]}"
echo "Total nodes: ${#NODELIST[@]}"

# ============================================================================
# Define node assignments
# ============================================================================
# Training nodes (16 total)
TRAIN_NODES="${NODELIST[0]},${NODELIST[1]},${NODELIST[2]},${NODELIST[3]},${NODELIST[5]},${NODELIST[6]},${NODELIST[7]},${NODELIST[8]},${NODELIST[10]},${NODELIST[11]},${NODELIST[12]},${NODELIST[13]},${NODELIST[15]},${NODELIST[16]},${NODELIST[17]},${NODELIST[18]}"

# vLLM nodes (4 total)
VLLM_NODE_0="${NODELIST[4]}"
VLLM_NODE_1="${NODELIST[9]}"
VLLM_NODE_2="${NODELIST[14]}"
VLLM_NODE_3="${NODELIST[19]}"

# Main process IP (first training node)
MAIN_PROCESS_IP="${NODELIST[0]}"

echo "=== Node Assignments ==="
echo "Training nodes: ${TRAIN_NODES}"
echo "vLLM nodes: ${VLLM_NODE_0}, ${VLLM_NODE_1}, ${VLLM_NODE_2}, ${VLLM_NODE_3}"
echo "Main process IP: ${MAIN_PROCESS_IP}"

# Export vLLM node addresses for training script
export VLLM_HOSTS="${VLLM_NODE_0}:8000,${VLLM_NODE_1}:8001,${VLLM_NODE_2}:8002,${VLLM_NODE_3}:8003"

# ============================================================================
# Start all 4 vLLM servers
# ============================================================================
echo "Starting vLLM servers..."

srun --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE_0}" --gres=gpu:4 \
     --output=logs/vllm_0_%j.out --error=logs/vllm_0_%j.err \
     trl vllm-serve --model Qwen/Qwen3-1.7B-Base --tensor_parallel_size 4 \
     --host 0.0.0.0 --port 8000 &

srun --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE_1}" --gres=gpu:4 \
     --output=logs/vllm_1_%j.out --error=logs/vllm_1_%j.err \
     trl vllm-serve --model Qwen/Qwen3-1.7B-Base --tensor_parallel_size 4 \
     --host 0.0.0.0 --port 8001 &

srun --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE_2}" --gres=gpu:4 \
     --output=logs/vllm_2_%j.out --error=logs/vllm_2_%j.err \
     trl vllm-serve --model Qwen/Qwen3-1.7B-Base --tensor_parallel_size 4 \
     --host 0.0.0.0 --port 8002 &

srun --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE_3}" --gres=gpu:4 \
     --output=logs/vllm_3_%j.out --error=logs/vllm_3_%j.err \
     trl vllm-serve --model Qwen/Qwen3-1.7B-Base --tensor_parallel_size 4 \
     --host 0.0.0.0 --port 8003 &

# Wait for vLLM servers to initialize
echo "Waiting for vLLM servers to initialize..."
sleep 90

# ============================================================================
# Start distributed training across all 16 training nodes
# ============================================================================
echo "Starting distributed training on 16 nodes (64 GPUs)..."

# Create a mapping file for nodes to their vLLM servers
cat > /tmp/vllm_mapping.txt << EOF
${NODELIST[0]}:${VLLM_NODE_0}:8000
${NODELIST[1]}:${VLLM_NODE_0}:8000
${NODELIST[2]}:${VLLM_NODE_0}:8000
${NODELIST[3]}:${VLLM_NODE_0}:8000
${NODELIST[5]}:${VLLM_NODE_1}:8001
${NODELIST[6]}:${VLLM_NODE_1}:8001
${NODELIST[7]}:${VLLM_NODE_1}:8001
${NODELIST[8]}:${VLLM_NODE_1}:8001
${NODELIST[10]}:${VLLM_NODE_2}:8002
${NODELIST[11]}:${VLLM_NODE_2}:8002
${NODELIST[12]}:${VLLM_NODE_2}:8002
${NODELIST[13]}:${VLLM_NODE_2}:8002
${NODELIST[15]}:${VLLM_NODE_3}:8003
${NODELIST[16]}:${VLLM_NODE_3}:8003
${NODELIST[17]}:${VLLM_NODE_3}:8003
${NODELIST[18]}:${VLLM_NODE_3}:8003
EOF

# Training node array for rank calculation
TRAIN_NODE_ARRAY=(${NODELIST[0]} ${NODELIST[1]} ${NODELIST[2]} ${NODELIST[3]} \
                  ${NODELIST[5]} ${NODELIST[6]} ${NODELIST[7]} ${NODELIST[8]} \
                  ${NODELIST[10]} ${NODELIST[11]} ${NODELIST[12]} ${NODELIST[13]} \
                  ${NODELIST[15]} ${NODELIST[16]} ${NODELIST[17]} ${NODELIST[18]})

srun --nodes=16 \
     --ntasks=16 \
     --nodelist="${TRAIN_NODES}" \
     --gres=gpu:4 \
     --output=logs/train_%j.out \
     --error=logs/train_%j.err \
     bash -c "
        # Get current hostname
        HOSTNAME=\$(hostname)
        
        # Calculate machine rank (0-15)
        MACHINE_RANK=0
        for i in \$(seq 0 15); do
            if [[ \"\$HOSTNAME\" == \"${TRAIN_NODE_ARRAY[\$i]}\"* ]]; then
                MACHINE_RANK=\$i
                break
            fi
        done
        
        # Get the vLLM server for this node from mapping
        VLLM_INFO=\$(grep \"^\$HOSTNAME\" /tmp/vllm_mapping.txt | cut -d: -f2,3)
        VLLM_HOST=\$(echo \$VLLM_INFO | cut -d: -f1)
        VLLM_PORT=\$(echo \$VLLM_INFO | cut -d: -f2)
        
        echo \"Node \$HOSTNAME: machine_rank=\$MACHINE_RANK, vllm=\$VLLM_HOST:\$VLLM_PORT\"
        
        accelerate launch \
            --config_file configs/multi_gpu.yaml \
            --num_processes 16 \
            --num_machines 16 \
            --main_process_ip ${MAIN_PROCESS_IP} \
            --main_process_port 29500 \
            --machine_rank \$MACHINE_RANK \
            train_grpo.py \
            --model_name Qwen/Qwen3-1.7B-Base \
            --use_vllm \
            --vllm_server_host \$VLLM_HOST:\$VLLM_PORT \
            --per_device_batch_size 1 \
            --sample_ratio 0.01 \
            --push_to_hub \
            --hub_username HuggingFaceAlbert \
            --output_dir ./outputs
     "

TRAIN_EXIT_CODE=$?

# ============================================================================
# Cleanup
# ============================================================================
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "Shutting down vLLM servers..."

# Kill all background jobs (vLLM servers)
jobs -p | xargs -r kill 2>/dev/null

echo "Job completed!"
exit $TRAIN_EXIT_CODE
