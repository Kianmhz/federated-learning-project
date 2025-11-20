# Federated Learning System for Edge Devices

A distributed machine learning system implementing **Federated Learning** to train models across multiple edge devices without sharing raw data. This project demonstrates key distributed systems principles including asynchronous communication, fault tolerance, privacy preservation, and real-time monitoring.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Team Members](#team-members)

---

## 🎯 Overview

### What is Federated Learning?

Federated Learning is a machine learning approach where:
- **Training happens on edge devices** (phones, IoT sensors, hospitals)
- **Only model updates are shared**, not raw data
- **Privacy is preserved** - sensitive data never leaves the device
- **Global model improves** through collaboration without centralization

### Problem Statement

Traditional machine learning requires centralizing data, which:
- ❌ Raises serious privacy concerns (GDPR, HIPAA)
- ❌ Uses massive bandwidth (transmitting raw data)
- ❌ Creates single points of failure
- ❌ Is insecure (data in transit and at rest)

### Our Solution

Federated Learning System that:
- ✅ Keeps data on edge devices (privacy preserved)
- ✅ Trains models locally (distributed computation)
- ✅ Shares only model updates (bandwidth efficient)
- ✅ Aggregates updates into global model (consensus protocol)
- ✅ Supports non-IID data (real-world conditions)
- ✅ Includes differential privacy (optional)
- ✅ Provides real-time monitoring (dashboard)

---

## ✨ Features

### Core Functionality
- **Federated Averaging (FedAvg)**: Weighted aggregation of client updates
- **Asynchronous Communication**: Clients work independently
- **Partial Participation**: Not all clients needed per round
- **Non-IID Data Distribution**: Realistic data heterogeneity
- **Differential Privacy**: Optional noise addition for privacy
- **Model Evaluation**: Automatic accuracy/loss tracking

### Technical Features
- **RESTful API**: FastAPI-based server
- **Real-time Dashboard**: React-based visualization
- **Scalable**: Support for 10-200+ clients
- **Fault Tolerant**: Handles client disconnections gracefully
- **Modular Design**: Clean separation of concerns

---

## 🏗️ System Architecture

Visualization of the system architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FEDERATED LEARNING SYSTEM                   │
└─────────────────────────────────────────────────────────────────┘

┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Client 1    │     │   Client 2    │     │   Client 3    │
│ ┌───────────┐ │     │ ┌───────────┐ │     │ ┌───────────┐ │
│ │Local Data │ │     │ │Local Data │ │     │ │Local Data │ │
│ │ 6000 img  │ │     │ │ 5800 img  │ │     │ │ 6200 img  │ │
│ └───────────┘ │     │ └───────────┘ │     │ └───────────┘ │
│       ↓       │     │       ↓       │     │       ↓       │
│ ┌───────────┐ │     │ ┌───────────┐ │     │ ┌───────────┐ │
│ │   Train   │ │     │ │   Train   │ │     │ │   Train   │ │
│ │  Locally  │ │     │ │  Locally  │ │     │ │  Locally  │ │
│ └───────────┘ │     │ └───────────┘ │     │ └───────────┘ │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        │   Model Updates     │                     │
        │   (HTTP POST)       │                     │
        └─────────────────────┼─────────────────────┘
                              ↓
                ┌──────────────────────────────┐
                │     Aggregation Server       │
                │  ┌────────────────────────┐  │
                │  │  Federated Averaging   │  │
                │  │  (Weighted Average)    │  │
                │  └────────────────────────┘  │
                │             ↓                │
                │  ┌────────────────────────┐  │
                │  │    Global Model        │  │
                │  │   (Improved Model)     │  │
                │  └────────────────────────┘  │
                └──────────────┬───────────────┘
                               │
                               ↓
                    ┌────────────────────┐
                    │   Dashboard (UI)   │
                    │  ┌──────────────┐  │
                    │  │ Accuracy     │  │
                    │  │ Loss         │  │
                    │  │ Clients      │  │
                    │  │ Rounds       │  │
                    │  └──────────────┘  │
                    └────────────────────┘
```

### Communication Flow

1. **Download Phase**: Clients download global model from server
2. **Training Phase**: Clients train locally on their data
3. **Upload Phase**: Clients send model updates (not raw data!)
4. **Aggregation Phase**: Server combines updates using FedAvg
5. **Evaluation Phase**: Server tests global model accuracy
6. **Monitoring Phase**: Dashboard polls server for metrics

---

## 📋 Prerequisites

### Software Requirements

- **Python**: 3.8 - 3.12 (recommended: 3.11)
- **Node.js**: 16.0+ (for React dashboard)
- **npm**: 8.0+ (comes with Node.js)
- **pip**: 21.0+ (for Python packages)

### Hardware Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: ~500MB for dataset and dependencies

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/Iqra-Z/federated-learning-project
cd federated-learning-project
```

### Step 2: Create Python Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- FastAPI (web framework)
- Uvicorn (ASGI server)
- PyTorch (machine learning)
- TorchVision (datasets)
- NumPy (data processing)
- Requests (HTTP client)


### Step 4: Setup React Dashboard

```bash
# Create React app
npx create-react-app dashboard

# Navigate to dashboard
cd dashboard

# Install dependencies
npm install recharts

# Copy the App.js code provided in artifacts
# (Replace dashboard/src/App.js with the enhanced version)

# Return to root
cd ..
```

---

## ⚡ Quick Start

### Terminal 1: Start Server

```bash
python server/main.py
```

**Expected output:**
```
======================================================================
🚀 FEDERATED LEARNING SERVER
======================================================================
📍 Server:    http://127.0.0.1:9000
📊 API Docs:  http://127.0.0.1:9000/docs
⚙️  Aggregation: Every 3 client updates
======================================================================
```

### Terminal 2-4: Start Clients

```bash
# Terminal 2 (Client 1)
python clients/client.py 1 --num-clients 10 --non-iid

# Terminal 3 (Client 2)
python clients/client.py 2 --num-clients 10 --non-iid

# Terminal 4 (Client 3)
python clients/client.py 3 --num-clients 10 --non-iid
```

**Expected output (each client):**
```
[CLIENT] Running with non-IID Dirichlet split (num_clients=10, alpha=0.5)
[CLIENT] Loaded partition for client 1: 6000 samples
[Client 1] Fetching global model...
[Client 1] Training locally...
[Client 1] Sending update...
```

### Terminal 5: Start Dashboard

```bash
cd dashboard
npm start
```

**Browser opens automatically at:** `http://localhost:3000`

---

## 📁 Project Structure

```
federated-learning-project/
│
├── server/
│   ├── __init__.py           # Package marker
│   ├── main.py              # FastAPI web server
│   └── aggregator.py        # FedAvg logic
│
├── clients/
│   ├── __init__.py           # Package marker
│   ├── client.py            # Client simulation
│   ├── data_utils.py        # Data partitioning
│   └── training.py          # Training loop
│
├── fl_core/
│   ├── __init__.py           # Package marker
│   └── model_def.py         # Model architecture
│
├── dashboard/                # React app
│   ├── src/
│   │   ├── App.js           # Main dashboard component
│   │   └── index.js         # React entry point
│   ├── public/
│   └── package.json
│
├── data/                     # Auto-created on first run
│   └── MNIST/               # Downloaded dataset
│
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

## 👥 Team Members

| Name | Role | Responsibilities |
|------|------|------------------|
| [Iqra Zahid], [Kianmehr Haddad Zahmatkesh] | ML Engineer & Project Lead | System design, integration, model training, dataset partitioning, FedAvg logic |
| [Abdulkarim Noorzaie], [AbdurRahman Abdurrahman] | Full-Stack Developer | Dashboard UI, API server, visualization, metrics tracking |

---

## 🙏 Acknowledgments

- **Instructor**: Dr. Khalid A. Hafeez
- **Course**: SOFE4790U - Distributed Systems (Fall 2025)
- **Institution**: Ontario Tech University
- **Team Members**: [Iqra Zahid], [Kianmehr Haddad Zahmatkesh], [Abdulkarim Noorzaie], [AbdurRahman Abdurrahman]

**Last Updated**: November 2025  
**Version**: 1.0.0
