### Top-Level Files & Directories
* **`main.py`**: The main experiment entry point. Handles command-line argument parsing, training scheduling, global weight aggregation (`average_weights`), logging, and result saving.
* **`README.md`**: Project documentation, including environment setup, usage examples, and experimental results.
* **`plots/`**: Contains Jupyter notebooks (`.ipynb`) and `plot_utils.py` for comprehensive data analysis and result visualization.
* **`maps/`**: Project mapping and auxiliary configuration files.
> *Note: The root directory also contains various non-code resources, such as network packet captures (`.pcapng`).*

---

### Core Codebase (`sim/`)
The `sim/` directory houses the core simulation logic, divided into four main modules:

#### 1. `algorithms/` (Learning Strategies)
* **`fedbase.py` & `fedbase2.py`**: Federated Learning base classes. Includes the core implementations and training/evaluation loops for `FedClient`, `FedGroup`, and `FedServer`.
* **`splitbase.py`**: Base classes designed for Split Learning methodologies.
* **`cwtbase.py`**: Base classes for CWT-related algorithmic variations.

#### 2. `data/` (Data Handling & Partitioning)
* **`data_utils.py`**: Dataset wrappers and data processing utility functions, including `FedDataset`.
* **`datasets.py` & `datasets_LR.py`**: Interfaces for constructing and loading various datasets (e.g., via `build_dataset`).
* **`partition.py`, `partition_LR.py`, `partition_SST2.py`, & `raw_partition/`**: Contains the logic for data partitioning and non-IID distribution strategies across clients (e.g., IID, Dirichlet, `group_dir`, `exdir`).

#### 3. `models/` (Neural Network Architectures)
* **`build_models.py`**: Factory functions to dynamically instantiate models by name (`build_model`).
* **`model_utils.py`**: General model-related utility functions.
* **Model Implementations**: Various network architectures tailored for specific datasets, such as `cifar_cnn.py`, `cifar_resnetii.py`, `mnist_wvgg.py`, `sst2_mlp.py`, and others.

#### 4. `utils/` (Helper Functions)
* **`optim_utils.py`**: Wrappers for optimizers and learning rate schedulers (`OptimKit`, `LrUpdater`).
* **`record_utils.py`**: Tools for experiment tracking, logging configuration, and result serialization (`record_exp_result`, `logconfig`).
* **`utils.py`**: General helper functions, such as initializing random seeds for reproducibility.



