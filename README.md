## Instructions

Here are the step-by-step instructions to set up your environment and run these three experiments.

## Catastrophic Forgetting Experiments

### Prerequisites

1.  **Hardware:** Apple Silicon M4. This code was run on a Macbook Pro using an M4 chip. For best reliability, using the same model is recomended.
2.  **Software:** Python 3.10 or higher.


### Step 1: Set up the Directory Structure
Download the files and navigate to the `catastrophic_forgetting_experiments` folder. You should see the following files.

```text
catastrophic_forgetting_experiments/
- Establishing_Catastrophic_Forgetting_Experiment.py
- Ablation_Experiment.py
- Sensitivity_Experiment.py
```

### Step 2: Install Dependencies

Open your terminal or command prompt, navigate to your folder, and install the required libraries.

```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes
```

For the GRPO experiments. Note you must have the same versions of the packages listed in the requirements.
```bash
pip install -r grpo_requirements.txt
```

### Step 3: Run the Experiments

You should run these scripts one by one. To run the experiments, run the following.

```bash
python Establishing_Catastrophic_Forgetting_Experiment.py
python Ablation_Experiment.py
python Sensitivity_Experiment.py
```

## Stabalization Experiments

### Navigate
Navigate to the `stabalization_experiments` folder. You should see the following files.

### Run
To run the GRPO naive baseline for coding task
```bash
python train_grpo_single.py --sample_ratio=1 --per_device_batch_size=28 --num_generations=4
```

To run our optimized baseline
```bash
python train_grpo_single_optimized.py --sample_ratio=1 --per_device_batch_size=28 --num_generations=4
```

