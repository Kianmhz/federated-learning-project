# ============ IMPORTS ============
import os
import sys  # Path manipulation
from datetime import datetime  # Timestamps

import torch  # PyTorch for model evaluation
from fastapi import FastAPI  # Web framework for building APIs
from fastapi.middleware.cors import CORSMiddleware  # Allow cross-origin requests
from pydantic import BaseModel  # Data validation

# Add parent directory to Python path (for imports to work)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.aggregator import Aggregator  # Your partner's aggregation logic

# ============ SETUP FASTAPI ============
app = FastAPI(title="Federated Learning Server", version="1.0.0")

# Enable CORS - Allow React dashboard (localhost:3000) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (fine for development)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# ============ GLOBAL STATE ============
# These variables persist across HTTP requests

agg = Aggregator()  # The aggregator (your partner's code)
clients_status = {}  # Track active clients: {client_id: {status, timestamp, ...}}
training_history = []  # Store metrics: [{round, accuracy, loss, ...}, ...]
UPDATES_PER_ROUND = 3  # Wait for 3 client updates before aggregating


# ============ DATA VALIDATION ============
class UpdateRequest(BaseModel):
    """Validate client update format"""

    client_id: str
    weights: dict  # Model weights as nested dict
    data_size: int  # Number of samples client trained on


# ============ API ENDPOINTS ============


@app.get("/")
def root():
    """
    Health check - Confirm server is running

    WHEN CALLED: Browser visits http://127.0.0.1:9000
    RETURNS: JSON with server status
    """
    return {
        "status": "Federated Learning Server Running",
        "round": agg.current_round,
        "clients_seen": len(clients_status),
        "pending_updates": len(agg.updates),
    }


@app.get("/get_model")
def get_model():
    """
    Clients download the global model

    WHEN CALLED: Client runs requests.get(f"{SERVER_URL}/get_model")
    WHAT IT DOES: Returns current global model weights
    RETURNS: {"round": 5, "weights": {...}}

    DISTRIBUTED CONCEPT: Model Synchronization
    All clients get same starting point each round
    """
    return {
        "round": agg.current_round,
        "weights": agg.get_weights(),  # Convert tensors → lists
    }


@app.post("/submit_update")
def submit_update(upd: UpdateRequest):
    """
    Clients submit trained model updates

    WHEN CALLED: Client runs requests.post(f"{SERVER_URL}/submit_update", json={...})
    WHAT IT DOES:
    1. Store client's update
    2. Track client activity
    3. If 3 updates received → aggregate them
    4. Evaluate new model accuracy
    5. Store metrics for dashboard

    RETURNS: {"status": "received", "round": 5, ...}

    DISTRIBUTED CONCEPT: Asynchronous Aggregation
    Clients submit independently, server aggregates when ready
    """
    # Store update in aggregator
    agg.receive_update(upd.weights, upd.data_size)

    # Track this client (for dashboard)
    client_id = upd.client_id
    clients_status[client_id] = {
        "status": "submitted",
        "timestamp": datetime.now().isoformat(),
        "data_size": upd.data_size,
        "round": agg.current_round,
    }

    print(
        f"[SERVER] ✓ Client {client_id} submitted. Updates: {len(agg.updates)}/{UPDATES_PER_ROUND}"
    )

    # Check if ready to aggregate
    if len(agg.updates) >= UPDATES_PER_ROUND:
        print(f"[SERVER] 🔄 Aggregating round {agg.current_round + 1}...")

        # Perform Federated Averaging (your partner's code)
        agg.aggregate()

        # Test the new model
        accuracy, loss = evaluate_model()

        # Store metrics for dashboard
        training_history.append(
            {
                "round": agg.current_round,
                "accuracy": accuracy,
                "loss": loss,
                "timestamp": datetime.now().isoformat(),
            }
        )

        print(
            f"[SERVER] ✅ Round {agg.current_round} complete. Acc: {accuracy:.2%}, Loss: {loss:.4f}"
        )

        return {
            "status": "aggregated",
            "round": agg.current_round,
            "accuracy": accuracy,
        }

    # Not enough updates yet
    return {
        "status": "received",
        "round": agg.current_round,
        "pending_updates": len(agg.updates),
    }


@app.get("/status")
def get_status():
    """
    Dashboard gets system status

    WHEN CALLED: Dashboard runs fetch('http://127.0.0.1:9000/status')
    WHAT IT DOES: Return snapshot of entire system state
    RETURNS: {current_round, active_clients, clients: {...}, history: [...]}

    USED BY: Dashboard to show real-time metrics
    """
    return {
        "current_round": agg.current_round,
        "active_clients": len(clients_status),
        "pending_updates": len(agg.updates),
        "clients": clients_status,
        "history": training_history[-20:],  # Last 20 rounds
    }


@app.get("/metrics")
def get_metrics():
    """
    Dashboard gets training metrics for charts

    WHEN CALLED: Dashboard runs fetch('http://127.0.0.1:9000/metrics')
    WHAT IT DOES: Return data formatted for charts
    RETURNS: {rounds: [...], accuracy: [...], loss: [...]}

    USED BY: Dashboard to plot accuracy/loss over time
    """
    return {
        "round": agg.current_round,
        "rounds": [h["round"] for h in training_history],
        "accuracy": [h["accuracy"] for h in training_history],
        "loss": [h["loss"] for h in training_history],
    }


# ============ HELPER FUNCTIONS ============


def evaluate_model():
    """
    Test global model on test dataset

    WHAT IT DOES:
    1. Load MNIST test set (10,000 images model hasn't seen)
    2. Run predictions on all images
    3. Calculate accuracy (% correct)
    4. Calculate loss (how confident/wrong predictions are)

    RETURNS: (accuracy, loss)

    WHY IMPORTANT: This tells us if model is actually learning!

    DISTRIBUTED CONCEPT: Centralized Evaluation
    Server evaluates on test set to measure global model quality
    """
    import torch.nn.functional as F

    from clients.data_utils import get_dataloader

    # Load test data
    _, test_loader = get_dataloader(batch_size=128)

    # Get model and set to evaluation mode
    model = agg.global_model
    model.eval()  # Disable dropout, batch norm

    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    # Test on all images
    with torch.no_grad():  # Don't compute gradients (faster)
        for images, labels in test_loader:
            outputs = model(images)

            # Calculate loss
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Count correct
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / num_batches

    return accuracy, avg_loss


# ============ START SERVER ============
if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("🚀 FEDERATED LEARNING SERVER")
    print("=" * 70)
    print(f"📍 Server:    http://127.0.0.1:9000")
    print(f"📊 API Docs:  http://127.0.0.1:9000/docs")
    print(f"⚙️  Aggregation: Every {UPDATES_PER_ROUND} client updates")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=9000)
