# Automating Data Governance: A Machine Learning Approach

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the paper: **"Automating Data Governance: A Machine Learning Approach for Lineage Tracking, Anomaly Detection, and Policy Enforcement in Enterprise Data Platforms"**

## Overview

This framework provides an integrated machine learning-based solution for automating enterprise data governance, combining:

1. **Lineage Inference Engine** - Graph Neural Network-based data lineage tracking
2. **Anomaly Detection Engine** - Deep autoencoder for detecting pipeline irregularities  
3. **Policy Enforcement Engine** - Automated compliance checking and enforcement

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Governance Automation Framework               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    Lineage      │  │    Anomaly      │  │     Policy      │ │
│  │   Inference     │  │   Detection     │  │   Enforcement   │ │
│  │    Engine       │  │    Engine       │  │     Engine      │ │
│  │                 │  │                 │  │                 │ │
│  │  • GNN-based    │  │  • Autoencoder  │  │  • Rule-based   │ │
│  │  • Metadata     │  │  • Statistical  │  │  • ML-enhanced  │ │
│  │    extraction   │  │    methods      │  │  • Real-time    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/example/data-governance-automation.git
cd data-governance-automation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from data_governance import GovernanceAutomationFramework
from data_governance.config import FrameworkConfig
from data_governance.synthetic_data import SyntheticDataGenerator

# Generate synthetic data
generator = SyntheticDataGenerator()
data = generator.generate_all()
normal_data, _ = generator.get_training_data()

# Initialize framework
config = FrameworkConfig()
framework = GovernanceAutomationFramework(config)
framework.initialize(training_data=normal_data)

# Process events
for event in data['processing_events']:
    result = framework.process_event(
        event_type=event['event_type'],
        source_system=event['source_system'],
        metadata=event
    )

# Get governance report
report = framework.get_governance_report()
print(f"Anomalies detected: {report['anomaly']['total_anomalies']}")
print(f"Policy violations: {report['policy']['total_violations']}")
```

## Running Experiments

```bash
# Run full experimental evaluation
python main.py --output outputs --num-events 10000 --anomaly-rate 0.05

# Run with custom parameters
python main.py -o results -n 50000 -a 0.03
```

## Project Structure

```
data-governance-automation/
├── src/
│   └── data_governance/
│       ├── __init__.py
│       ├── config.py          # Configuration classes
│       ├── lineage.py         # Lineage inference engine
│       ├── anomaly.py         # Anomaly detection engine
│       ├── policy.py          # Policy enforcement engine
│       ├── framework.py       # Integrated framework
│       ├── synthetic_data.py  # Data generation
│       └── evaluation.py      # Evaluation metrics
├── tests/
│   └── test_framework.py      # Unit tests
├── main.py                    # Experiment runner
├── requirements.txt
├── setup.py
└── README.md
```

## Key Components

### 1. Lineage Inference Engine

Uses Graph Attention Networks (GAT) to infer data lineage relationships:

```python
# Node embedding update (Equation from Section 3.2)
# h_v^(k) = σ(W^(k) · AGG({h_u^(k-1) : u ∈ N(v)}))
```

### 2. Anomaly Detection Engine

Deep autoencoder architecture for detecting anomalies:

- Encoder: 256 → 128 → 64 → 32 (latent)
- Decoder: 32 → 64 → 128 → 256
- Anomaly threshold τ based on reconstruction error percentile

### 3. Policy Enforcement Engine

Policy evaluation function:

```python
# Policy activates when: P_anomaly(context) AND C_compliance(context)
```

## Evaluation Results

Based on experimental evaluation (Section 5):

| Metric | Proposed (ML-Driven) | Rule-Based Baseline |
|--------|---------------------|---------------------|
| Lineage Precision | 0.94 | 0.78 |
| Lineage Recall | 0.91 | 0.72 |
| Lineage F1 | 0.92 | 0.75 |
| Anomaly F1 | 0.88 | 0.81 |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/data_governance --cov-report=html
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{datagovernance2025,
  title={Automating Data Governance: A Machine Learning Approach for 
         Lineage Tracking, Anomaly Detection, and Policy Enforcement 
         in Enterprise Data Platforms},
  author={Kaithapuram, Vishnuvardhan Reddy},
  journal={...},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
