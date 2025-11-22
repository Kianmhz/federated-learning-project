# ============ IMPORTS ============
import os
import sys  # Path manipulation
from datetime import datetime  # Timestamps
from math import ceil
import random  # Random sampling

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

# Partial participation settings
# p = probability a client is selected each round (server-side configured)
# expected_selected = round(p * num_clients) can be used as a target, but server
# will aggregate as soon as it receives updates from all selected clients OR when
# at least MIN_UPDATES_TO_AGGREGATE are available.
PARTICIPATION_PROB = 0.5
NUM_CLIENTS = 10
MIN_UPDATES_TO_AGGREGATE = 1  # safety lower bound
UPDATES_PER_ROUND = max(MIN_UPDATES_TO_AGGREGATE, ceil(PARTICIPATION_PROB * NUM_CLIENTS))

# Deterministic per-round selection storage: round_index -> set(client_id_str)
selected_clients_by_round = {}

# Adaptive participation tuning parameters
# If recent accuracy improvement over P_ADJUST_WINDOW rounds is < P_IMPROVE_MIN,
# increase PARTICIPATION_PROB by P_INCREMENT (once per round at most).
P_ADJUST_WINDOW = 5
P_IMPROVE_MIN = 1e-3
P_INCREMENT = 0.1
last_p_adjust_round = -1


def update_updates_per_round():
    """Update UPDATES_PER_ROUND from PARTICIPATION_PROB and NUM_CLIENTS."""
    global UPDATES_PER_ROUND
    UPDATES_PER_ROUND = max(MIN_UPDATES_TO_AGGREGATE, ceil(PARTICIPATION_PROB * NUM_CLIENTS))


def select_participants_for_round(round_idx: int):
    """Select and persist a deterministic participant set for the given round.

    Selection draws k = round(PARTICIPATION_PROB * NUM_CLIENTS) unique client ids
    from the range [0, NUM_CLIENTS). Returned client ids are strings to match
    client_id usage elsewhere.
    """
    if round_idx in selected_clients_by_round:
        return selected_clients_by_round[round_idx]

    k = max(1, int(round(PARTICIPATION_PROB * NUM_CLIENTS)))
    k = min(k, NUM_CLIENTS)
    sampled = set(str(x) for x in random.sample(range(NUM_CLIENTS), k))
    selected_clients_by_round[round_idx] = sampled
    print(f"[SERVER] Selected {len(sampled)} participants for round {round_idx}: {sorted(list(sampled))}")
    return sampled


def adjust_participation():
    """Increase PARTICIPATION_PROB when convergence stalls over recent rounds.

    Simple heuristic: if improvement over the last P_ADJUST_WINDOW rounds is less
    than P_IMPROVE_MIN, bump PARTICIPATION_PROB by P_INCREMENT (capped at 1.0)
    and recompute UPDATES_PER_ROUND. Only adjust once per round.
    """
    global PARTICIPATION_PROB, last_p_adjust_round
    current = agg.current_round
    if current <= 0:
        return
    if last_p_adjust_round >= current:
        return
    if len(training_history) < P_ADJUST_WINDOW:
        return

    recent = training_history[-P_ADJUST_WINDOW:]
    first_acc = recent[0]["accuracy"]
    last_acc = recent[-1]["accuracy"]
    improvement = last_acc - first_acc
    if improvement < P_IMPROVE_MIN and PARTICIPATION_PROB < 1.0:
        PARTICIPATION_PROB = min(1.0, PARTICIPATION_PROB + P_INCREMENT)
        update_updates_per_round()
        last_p_adjust_round = current
        print(f"[SERVER] ↑ Participation increased to {PARTICIPATION_PROB:.2f}; target updates {UPDATES_PER_ROUND}")


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


@app.get("/should_participate")
def should_participate(client_id: str):
    """
    Clients can call this endpoint to check whether they were randomly selected
    to participate in the current round. Server uses PARTICIPATION_PROB to
    select clients per round.

    Returns: {selected: bool, round: int}
    """

    # Use deterministic per-round selection: server picks participants once per
    # round and records them in `selected_clients_by_round`.
    selected_set = select_participants_for_round(agg.current_round)
    selected = str(client_id) in selected_set
    return {"selected": selected, "round": agg.current_round}


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

        # Optionally adjust participation probability based on recent progress
        try:
            adjust_participation()
        except Exception as e:
            print(f"[SERVER] Participation adjustment failed: {e}")

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
