# MLOpsLib

## Description
MLOpsLib is a MLDevOps Library designed for easy configuration via yaml files, with a modular architecture, checkpoint system and class registry


## Key Features

- Full Pipeline configuration via yaml files
- Modular Architecture (Data Ingestion, Train Test Split, Data Preparation, Model Training, Model Evaluation)
- Checkpoint System for reduced computational load
- Run Tracking with MLFlow
- Registry to add new classes 
- Support for sklearn and pytorch models

## Installation

### Prerequisites
- Python 3.10 or higher

### Install from Source

Clone this repository and install it with `pip`:

```bash
git clone https://github.com/JannisEbling/MLOpsLib.git
cd MLOps
pip install -r requirements.txt