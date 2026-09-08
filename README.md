# ChemNetwork

ChemNetwork is a small research and learning project focused on building a practical ML application stack around a generative modeling idea. The repository brings together a FastAPI backend, database integration, experiment tracking, and a toy flow-matching model to explore how these components fit into a real project workflow.

## Overview

This project is primarily a sandbox for learning and experimenting with:

- FastAPI and API design
- pytest-based backend testing
- Dockerized development and deployment patterns
- database-backed application structure
- generative modeling with a minimal working implementation

> [!NOTE]
> The frontend, especially the styling in the UI, was created with the help of generative AI. Since the main goal of this project is to learn the backend and ML stack, frontend work remains secondary and is mainly used for lightweight testing and demonstration.

## Current Model

The backend currently includes a toy flow-matching model, a generative approach that learns a vector field to move samples from a simple distribution toward a more complex target distribution through integration. This follows the flow matching framework introduced in the paper [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) by Lipman et al.

## Tech Stack

- Python
- FastAPI
- PyTorch
- pytest
- Docker
- MongoDB / database-backed application patterns
- YAML-based configuration and experiment management

## Quick Start

First, if you'd like to run and develop this project locally, make sure you have uv installed on your system. Find an installation guide [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Create virtural environment and install dependencies locally
uv sync
```

To start the app, simply start the docker container
```
# Start the docker container
docker compose up -d
```

So far, the container only runs locally, and it can be addressed from your machine via *port 8000*. That is, to open the web client, open (localhost:8000)[http://localhost:8000/].
You can find the SwaggerUI style docs under (http://localhost:8000/docs)[localhost:8000/docs].

For containerized development, use the provided Docker configuration in the repository root.

## Repository Structure

```text
.
├── app/                          # FastAPI application and backend logic
│   ├── db/                      # Database setup and experiments
│   ├── routers/                 # API endpoints
│   ├── static/                  # Frontend assets and UI files
│   ├── main.py                  # Application entry point
│   ├── config.py                # App configuration
│   ├── dependencies.py          # Dependency injection and shared setup
│   ├── schemas.py               # Request/response models
│   └── serialization.py         # Data serialization helpers
├── chemnetwork/                 # Core model and data science code
│   ├── data/                    # Data utilities and sample datasets
│   ├── models/                  # Model implementations and notebooks
│   ├── training/                # Training scripts and training logic
│   ├── sample.py                # Sample/demo code for model usage
│   └── utils.py                 # Shared utility functions
├── preferences/                 # YAML configs and experiment parameters
├── runs/                        # Training outputs, checkpoints, and run metadata
├── tests/                       # Unit tests for app and model behavior
├── docker-compose.yaml          # Docker Compose setup
├── Dockerfile                   # Container build definition
├── pyproject.toml               # Project dependencies and tooling
├── README.md                    # Project overview and documentation
├── TODO.md                      # Current tasks and ideas
├── serving_checkpoint/          # Pretrained model checkpoint artifacts
└── wandb/                       # Weights & Biases run metadata and outputs
```

## Status

The repository is currently in an exploratory phase. The focus is to validate the stack, understand the model workflow, and build a solid foundation before moving to a real dataset and a meaningful chemical or bioinformatics task.

## Long-Term Direction

The long-term goal is to work with a real dataset and apply the system to a concrete problem such as chemical structure generation, structure classification, or another chemistry-related ML task.
