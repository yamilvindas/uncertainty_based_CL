# **Uncertainty-Based Replay Continual Learning for Medical Imaging**

This repository implements an advanced uncertainty-based sample selection strategy for replay-based Continual Learning (CL) in medical imaging.  
The framework is evaluated across three primary datasets:

1. **OrganMNIST** (Public)  
2. **Camelyon17** (Public \- WILDS)  
3. **Transcranial Doppler Ultrasounds (TCDs)** (Private)

## **I. Environment Preparation**

We recommend using a virtual environment to manage dependencies cleanly.

1. **Create the virtual environment:**  
   python \-m venv .venv

2. **Activate the environment:**  
   * *Linux/macOS:*  
     source .venv/bin/activate

   * *Windows:*  
     .venv\\Scripts\\activate

3. **Install the requirements:**  
   python \-m pip install \--upgrade pip  
   pip install \-r requirements.txt

## **II. Code Structure**

The repository is organized into three main directories: src/, configs/, and exec\_files/.

* **src/**: Contains all the source code for the experiments.  
  * **continual\_learning/**: Core Continual Learning mechanisms.  
    * *ewc.py*: Implements Elastic Weight Consolidation (computes Fisher Information Matrix and EWC penalty).  
    * *memory.py*: Manages the replay memory buffer, including latent space plotting (t-SNE) and various sample selection strategies (Uniform, Uncertainty-based (Proposed), Feature Dissimilarity, Loss-based).  
  * **data\_processing/**: Data handlers and preprocessing pipelines.  
    * *OrganMNIST.py*: Task A (OrganAMNIST) vs. Task B (OrganCMNIST).  
    * *Camelyon17.py*: Task A (Hospitals 1 & 2), Task B (Hospitals 3 & 4), and External Validation (Hospital 5).  
  * **experiments/**:  
    * *main\_experiment.py*: The primary entry point. Executes OPTUNA hyperparameter optimization followed by the full repeated-holdout evaluation.  
  * **models/**: Neural network architectures.  
    * *simple\_cnn.py*: Lightweight custom CNNs (e.g., tailored CNN for OrganMNIST, MobileNetV3 for Camelyon17).  
    * *resnet.py*: Standard pre-trained ResNet18-based model.  
  * **utils/**:  
    * *analyze\_results.py*: Extracts hard predictions and probabilities from HDF5 files to compute statistical metrics (Balanced Acc, MCC, F1-Score, AUC-ROC) and quantify catastrophic forgetting.  
    * *config\_generator.py*: PGenerates the YAML configuration trees for all experimental variations.  
* **configs/**: Stores the generated YAML parameter files. *(Populated using config\_generator.py)*  
* **exec\_files/**: Contains bash scripts for orchestrating batch runs, including fault-tolerant resume capabilities.

## **III. Usage**

### **A. (Re-)Generate Configuration Files**

Before running experiments, generate the hierarchical configuration files for OrganMNIST and Camelyon17:  
python src/utils/config\_generator.py

If you also want to generate configuration files that explicitly combine EWC with replay-based CL, append the flag:  
python src/utils/config\_generator.py \--generate\_EWC\_Replay\_combination

If you want to use a HITS dataset, you need to specify the paths to the hdf5 files describing the splits:
python src/utils/config\_generator.py --HITS-data-path-A path/to/data_A.hdf5 --HITS-data-path-B path/to/data_B.hdf5

### **B. Launch Experiments**

**To launch a single, isolated experiment:**  
Pass the target configuration YAML directly to the main script:  
python src/experiments/main\_experiment.py \--parameters\_file configs/OrganMNIST/Baseline/EWC.yaml

Each completed experiment generates a structured folder inside ./results/EXP-ID/, containing:

* 📁 **memories/**: Contains 2D t-SNE projections showing the memory buffer distributed across the full training manifold (empty if no replay strategy is used).  
* 📁 **metrics/**: Contains the OPTUNA .db tracking file and predictions_i.h5. The HDF5 file logs all model predictions, targets, and probabilities across training phases.  
* 📁 **models/**: Stores the serialized .pt weights for all models generated during the repeated holdouts.

**Note*:* To run any of the continual learning experiments, you must first run a full baseline experiment (without memory and EWC), as all continual learning experiments use the baseline as a starting point.

**To run all experiments sequentially (Batch Mode):**  
Use the provided bash runner. It includes a resume feature, meaning if the process is interrupted, re-running the command will skip completed experiments and pick up right where it left off.  
bash exec\_files/experiment\_runner.bash

### **C. Analyze Results**

You can compute the mean, standard deviation, and quantify catastrophic forgetting statistics for any completed experiment using the analysis utility:  
python src/utils/analyze\_results.py results/EXP-RESULTS-FOLDER/metrics/predictions_i.h5

*(Note: Ensure you point the script to the actual .h5 file generated in your results directory).*
