import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

const SERVER_URL = "http://127.0.0.1:9000";

function App() {
  // ============================================================
  // STATE - Data that triggers re-renders when changed
  // ============================================================

  // System status: rounds, clients, pending updates
  const [status, setStatus] = useState({
    current_round: 0,
    active_clients: 0,
    pending_updates: 0,
    clients: {},
    history: [],
  });

  // Training metrics: accuracy and loss arrays
  const [metrics, setMetrics] = useState({
    rounds: [],
    accuracy: [],
    loss: [],
  });

  // UI state: loading and error handling
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    // Fetch immediately on mount
    fetchStatus();
    fetchMetrics();

    // Set up polling (fetch every X seconds)
    const statusInterval = setInterval(fetchStatus, 2000); // Every 2s
    const metricsInterval = setInterval(fetchMetrics, 5000); // Every 5s

    // Cleanup when component unmounts
    return () => {
      clearInterval(statusInterval);
      clearInterval(metricsInterval);
    };
  }, []); // Empty array = run once on mount

  // ============================================================
  // FETCH FUNCTIONS - Async/Await for HTTP Requests
  // ============================================================

  const fetchStatus = async () => {
    try {
      // Send GET request to server
      const response = await fetch(`${SERVER_URL}/status`);

      // Check if request succeeded (status 200)
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      // Parse JSON response
      const data = await response.json();

      // Update state (triggers re-render)
      setStatus(data);
      setLastUpdate(new Date());
      setIsLoading(false);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch status:", err);
      setError(err.message);
      setIsLoading(false);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/metrics`);
      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      console.error("Failed to fetch metrics:", err);
    }
  };

  // ============================================================
  // DATA TRANSFORMATION - Prepare data for charts
  // ============================================================

  /*
  RECHARTS needs data as array of objects:
  [
    {round: 1, accuracy: 85, loss: 0.45},
    {round: 2, accuracy: 88, loss: 0.38},
    ...
  ]

  We have parallel arrays from server:
  - rounds: [1, 2, 3]
  - accuracy: [0.85, 0.88, 0.91]
  - loss: [0.45, 0.38, 0.32]

  ZIP them together using .map():
  */
  const chartData = metrics.rounds.map((round, idx) => ({
    round,
    accuracy: parseFloat((metrics.accuracy[idx] * 100).toFixed(2)),
    loss: parseFloat(metrics.loss[idx].toFixed(4)),
  }));

  // Latest accuracy (for display)
  const latestAccuracy =
    chartData.length > 0 ? chartData[chartData.length - 1].accuracy : 0;

  // ============================================================
  // LOADING & ERROR STATES
  // ============================================================

  // Show loading screen while fetching initial data
  if (isLoading) {
    return (
      <div style={styles.loadingContainer}>
        <div style={styles.spinner}>⏳</div>
        <div style={styles.loadingText}>Connecting to server...</div>
      </div>
    );
  }

  // Show error screen if connection failed
  if (error) {
    return (
      <div style={styles.errorContainer}>
        <h2>❌ Connection Error</h2>
        <p>
          <strong>Could not connect to server:</strong> {error}
        </p>
        <p>Make sure the server is running:</p>
        <code style={styles.codeBlock}>python server/main.py</code>
      </div>
    );
  }

  // ============================================================
  // MAIN RENDER - Dashboard UI
  // ============================================================

  return (
    <div style={styles.container}>
      {/* Header with gradient */}
      <div style={styles.header}>
        <h1 style={styles.title}>
          <span style={styles.emoji}>🤖</span> Federated Learning Dashboard
        </h1>
        <div style={styles.subtitle}>
          Real-time Distributed Machine Learning Visualization
        </div>
        {lastUpdate && (
          <div style={styles.lastUpdate}>
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Top Stats Row - 4 cards */}
      <div style={styles.statsGrid}>
        <StatCard
          title="Current Round"
          value={status.current_round}
          color="#4CAF50"
          icon="🔄"
          subtitle="Training iterations"
        />
        <StatCard
          title="Active Clients"
          value={status.active_clients}
          color="#2196F3"
          icon="📱"
          subtitle="Edge devices"
        />
        <StatCard
          title="Pending Updates"
          value={status.pending_updates}
          color="#FF9800"
          icon="⏳"
          subtitle="Awaiting aggregation"
        />
        <StatCard
          title="Model Accuracy"
          value={`${latestAccuracy.toFixed(1)}%`}
          color="#9C27B0"
          icon="🎯"
          subtitle="Test set performance"
        />
      </div>

      {/* Charts Row - Two charts side by side */}
      <div style={styles.chartsGrid}>
        {/* Accuracy Chart - Area Chart */}
        <ChartCard title="📈 Model Accuracy Over Rounds" color="#4CAF50">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient
                    id="colorAccuracy"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="5%" stopColor="#4CAF50" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#4CAF50" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  dataKey="round"
                  label={{
                    value: "Round",
                    position: "insideBottom",
                    offset: -5,
                  }}
                  stroke="#666"
                />
                <YAxis
                  label={{
                    value: "Accuracy (%)",
                    angle: -90,
                    position: "insideLeft",
                  }}
                  domain={[0, 100]}
                  stroke="#666"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#fff",
                    border: "1px solid #ddd",
                    borderRadius: "4px",
                  }}
                  formatter={(value) => `${value}%`}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#4CAF50"
                  fillOpacity={1}
                  fill="url(#colorAccuracy)"
                  strokeWidth={3}
                  name="Accuracy (%)"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState message="Waiting for training data..." icon="📊" />
          )}
        </ChartCard>

        {/* Loss Chart - Line Chart */}
        <ChartCard title="📉 Training Loss Over Rounds" color="#F44336">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  dataKey="round"
                  label={{
                    value: "Round",
                    position: "insideBottom",
                    offset: -5,
                  }}
                  stroke="#666"
                />
                <YAxis
                  label={{ value: "Loss", angle: -90, position: "insideLeft" }}
                  stroke="#666"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#fff",
                    border: "1px solid #ddd",
                    borderRadius: "4px",
                  }}
                  formatter={(value) => value.toFixed(4)}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#F44336"
                  strokeWidth={3}
                  dot={{ r: 5, fill: "#F44336" }}
                  activeDot={{ r: 8 }}
                  name="Cross-Entropy Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState message="Waiting for training data..." icon="📊" />
          )}
        </ChartCard>
      </div>

      {/* Active Clients Table */}
      <div style={styles.tableCard}>
        <h2 style={styles.cardTitle}>
          📊 Active Clients
          <span style={styles.badge}>
            {Object.keys(status.clients).length} connected
          </span>
        </h2>

        {Object.keys(status.clients).length === 0 ? (
          <EmptyState
            message="No clients connected yet. Start some clients to see them here!"
            icon="🤖"
          />
        ) : (
          <div style={styles.tableWrapper}>
            <table style={styles.table}>
              <thead>
                <tr style={styles.tableHeader}>
                  <th style={styles.th}>Client ID</th>
                  <th style={styles.th}>Status</th>
                  <th style={styles.th}>Data Size</th>
                  <th style={styles.th}>Round</th>
                  <th style={styles.th}>Last Update</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(status.clients).map(
                  ([clientId, clientInfo]) => (
                    <tr key={clientId} style={styles.tableRow}>
                      <td style={styles.td}>
                        <span
                          style={{
                            ...styles.clientBadge,
                            backgroundColor: "#4CAF50",
                          }}
                        >
                          Client {clientId}
                        </span>
                      </td>
                      <td style={styles.td}>
                        <StatusBadge status={clientInfo.status} />
                      </td>
                      <td style={styles.td}>
                        <strong>{clientInfo.data_size.toLocaleString()}</strong>{" "}
                        samples
                      </td>
                      <td style={styles.td}>
                        <span style={styles.roundBadge}>
                          Round {clientInfo.round}
                        </span>
                      </td>
                      <td style={styles.td}>
                        {new Date(clientInfo.timestamp).toLocaleTimeString()}
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Instructions Panel */}
      <div style={styles.instructionsCard}>
        <h3 style={styles.instructionsTitle}>💡 Quick Start Guide</h3>
        <div style={styles.instructionsGrid}>
          <div style={styles.instructionStep}>
            <div style={styles.stepNumber}>1</div>
            <div style={styles.stepContent}>
              <strong>Start Server</strong>
              <code style={styles.inlineCode}>python server/main.py</code>
            </div>
          </div>
          <div style={styles.instructionStep}>
            <div style={styles.stepNumber}>2</div>
            <div style={styles.stepContent}>
              <strong>Launch Clients</strong>
              <code style={styles.inlineCode}>
                python clients/client.py 1 --num-clients 10 --non-iid
              </code>
            </div>
          </div>
          <div style={styles.instructionStep}>
            <div style={styles.stepNumber}>3</div>
            <div style={styles.stepContent}>
              <strong>Monitor Training</strong>
              <span>Watch metrics update in real-time!</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, value, color, icon, subtitle }) {
  return (
    <div style={{ ...styles.statCard, borderLeftColor: color }}>
      <div style={styles.statIcon}>{icon}</div>
      <div style={styles.statContent}>
        <div style={styles.statTitle}>{title}</div>
        <div style={{ ...styles.statValue, color }}>{value}</div>
        {subtitle && <div style={styles.statSubtitle}>{subtitle}</div>}
      </div>
    </div>
  );
}

function ChartCard({ title, children, color }) {
  return (
    <div style={styles.chartCard}>
      <h2
        style={{
          ...styles.cardTitle,
          borderLeftColor: color,
          paddingLeft: "15px",
        }}
      >
        {title}
      </h2>
      <div style={styles.chartContent}>{children}</div>
    </div>
  );
}

function StatusBadge({ status }) {
  const config = {
    submitted: { color: "#4CAF50", label: "✓ Submitted" },
    training: { color: "#FF9800", label: "⚙ Training" },
    idle: { color: "#9E9E9E", label: "○ Idle" },
    error: { color: "#F44336", label: "✗ Error" },
  }[status] || { color: "#9E9E9E", label: status };

  return (
    <span
      style={{
        backgroundColor: config.color,
        color: "white",
        padding: "4px 12px",
        borderRadius: "12px",
        fontSize: "12px",
        fontWeight: "bold",
      }}
    >
      {config.label}
    </span>
  );
}

function EmptyState({ message, icon = "📭" }) {
  return (
    <div style={styles.emptyState}>
      <div style={styles.emptyIcon}>{icon}</div>
      <p>{message}</p>
    </div>
  );
}

// ============================================================
// STYLES - CSS in JavaScript
// ============================================================

const styles = {
  container: {
    padding: "20px",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    backgroundColor: "#f5f7fa",
    minHeight: "100vh",
    maxWidth: "1400px",
    margin: "0 auto",
  },
  header: {
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    padding: "30px",
    borderRadius: "12px",
    marginBottom: "30px",
    boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
    color: "white",
  },
  title: {
    margin: 0,
    fontSize: "32px",
    fontWeight: "700",
    display: "flex",
    alignItems: "center",
    gap: "15px",
  },
  emoji: {
    fontSize: "40px",
  },
  subtitle: {
    fontSize: "16px",
    opacity: 0.9,
    marginTop: "10px",
  },
  lastUpdate: {
    fontSize: "14px",
    opacity: 0.8,
    marginTop: "10px",
  },
  statsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
    gap: "20px",
    marginBottom: "30px",
  },
  statCard: {
    backgroundColor: "white",
    padding: "25px",
    borderRadius: "12px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.08)",
    borderLeft: "4px solid",
    display: "flex",
    alignItems: "center",
    gap: "20px",
    transition: "transform 0.2s, box-shadow 0.2s",
    cursor: "default",
  },
  statIcon: {
    fontSize: "40px",
  },
  statContent: {
    flex: 1,
  },
  statTitle: {
    fontSize: "14px",
    color: "#666",
    marginBottom: "5px",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  },
  statValue: {
    fontSize: "32px",
    fontWeight: "bold",
    marginBottom: "5px",
  },
  statSubtitle: {
    fontSize: "12px",
    color: "#999",
  },
  chartsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(500px, 1fr))",
    gap: "20px",
    marginBottom: "30px",
  },
  chartCard: {
    backgroundColor: "white",
    padding: "20px",
    borderRadius: "12px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.08)",
  },
  cardTitle: {
    marginTop: 0,
    color: "#333",
    marginBottom: "20px",
    fontSize: "18px",
    fontWeight: "600",
    borderLeft: "4px solid",
    paddingLeft: "15px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  badge: {
    fontSize: "14px",
    backgroundColor: "#e3f2fd",
    color: "#2196F3",
    padding: "4px 12px",
    borderRadius: "12px",
    fontWeight: "normal",
  },
  chartContent: {
    minHeight: "300px",
  },
  tableCard: {
    backgroundColor: "white",
    padding: "20px",
    borderRadius: "12px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.08)",
    marginBottom: "30px",
  },
  tableWrapper: {
    overflowX: "auto",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
  },
  tableHeader: {
    backgroundColor: "#f8f9fa",
    textAlign: "left",
  },
  th: {
    padding: "15px",
    borderBottom: "2px solid #dee2e6",
    fontWeight: "600",
    color: "#495057",
    fontSize: "14px",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  },
  tableRow: {
    borderBottom: "1px solid #e9ecef",
    transition: "background-color 0.2s",
  },
  td: {
    padding: "15px",
    fontSize: "14px",
    color: "#495057",
  },
  clientBadge: {
    color: "white",
    padding: "6px 12px",
    borderRadius: "6px",
    fontSize: "13px",
    fontWeight: "600",
  },
  roundBadge: {
    backgroundColor: "#e3f2fd",
    color: "#2196F3",
    padding: "4px 10px",
    borderRadius: "12px",
    fontSize: "12px",
    fontWeight: "600",
  },
  instructionsCard: {
    backgroundColor: "#e3f2fd",
    padding: "25px",
    borderRadius: "12px",
    borderLeft: "4px solid #2196F3",
  },
  instructionsTitle: {
    marginTop: 0,
    color: "#1976d2",
    fontSize: "20px",
    marginBottom: "20px",
  },
  instructionsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
    gap: "20px",
  },
  instructionStep: {
    display: "flex",
    alignItems: "flex-start",
    gap: "15px",
  },
  stepNumber: {
    backgroundColor: "#2196F3",
    color: "white",
    width: "32px",
    height: "32px",
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "bold",
    flexShrink: 0,
  },
  stepContent: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  inlineCode: {
    backgroundColor: "#fff",
    padding: "4px 8px",
    borderRadius: "4px",
    fontSize: "13px",
    fontFamily: "monospace",
    color: "#1976d2",
    display: "block",
  },
  emptyState: {
    textAlign: "center",
    padding: "60px 20px",
    color: "#999",
  },
  emptyIcon: {
    fontSize: "48px",
    marginBottom: "15px",
  },
  loadingContainer: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    height: "100vh",
    backgroundColor: "#f5f7fa",
  },
  spinner: {
    fontSize: "64px",
    marginBottom: "20px",
    animation: "spin 2s linear infinite",
  },
  loadingText: {
    fontSize: "24px",
    color: "#666",
  },
  errorContainer: {
    padding: "40px",
    backgroundColor: "#ffebee",
    color: "#c62828",
    borderRadius: "12px",
    margin: "40px",
    maxWidth: "600px",
  },
  codeBlock: {
    display: "block",
    backgroundColor: "#fff",
    padding: "15px",
    marginTop: "15px",
    color: "#000",
    borderRadius: "6px",
    fontFamily: "monospace",
  },
};

export default App;
