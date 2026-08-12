# ChemNetwork

My first FastAPI project. Goal: build a neural network to predict chemical structure properties using data from PubChem/ChEMBL.

### train a model
python -m chemnetwork.training.train train_parameters.comment=test1

### health check (defined in main.py)
curl http://localhost:8000/health

### generate samples — POST /generate with a JSON body
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 10, "integration_steps": 100}'