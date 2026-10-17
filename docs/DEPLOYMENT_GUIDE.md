# Local AI Assistant Deployment & Research Architecture
**Host Environment:** Apple Silicon M4, macOS, 24 GB Unified Memory  
**Public Ingress Domain:** `https://ai.iliev.us` (Cloudflare Tunnel `mac-ai`)  
**Base Directory:** `/Users/Sergiu/AI`

---

## 1. System Topology & Security Architecture

```
                  [ Web Browser / Internet Client ]
                                  │
                                  ▼ (Encrypted HTTPS Edge TLS)
                   ┌─────────────────────────────┐
                   │   Cloudflare Zero Trust     │
                   │   Hostname: ai.iliev.us     │
                   │   Tunnel ID: mac-ai         │
                   └──────────────┬──────────────┘
                                  │ (Encrypted QUIC Argo Tunnel)
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Apple Silicon M4 Workstation (Local Sandbox: /Users/Sergiu/AI)   │
│                                                                 │
│  ┌─────────────────────────┐                                    │
│  │ cloudflared Connector   │                                    │
│  │ (Proxies traffic to     │                                    │
│  │  127.0.0.1:8080 only)   │                                    │
│  └────────────┬────────────┘                                    │
│               │ (Loopback only)                                 │
│               ▼                                                 │
│  ┌─────────────────────────┐                                    │
│  │ Open WebUI (FastAPI)    │ (Port 8080, Auth Enforced)         │
│  │ DB: data/open-webui/    │                                    │
│  └────────────┬────────────┘                                    │
│               │ (Loopback only)                                 │
│               ▼                                                 │
│  ┌─────────────────────────┐      ┌───────────────────────────┐ │
│  │ Ollama Inference Engine │      │ Heretic Research Sandbox  │ │
│  │ Port: 127.0.0.1:11434   │      │ (PyTorch MPS / Metal)     │ │
│  │ Metal GPU Accelerated   │      │ research/heretic/         │ │
│  │ Models: gemma2:2b,      │      │ Refusal vector ablation & │ │
│  │         llama3.2:3b     │      │ representation audit      │ │
│  └─────────────────────────┘      └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Security & Ingress Invariants:
1. **Zero External Port Forwarding**: No router ports (NAT) are opened. All ingress is proxied through an encrypted outbound Cloudflare QUIC tunnel.
2. **API Non-Exposure**: The local Ollama inference API (`127.0.0.1:11434`) is bound strictly to `localhost` and is NEVER exposed to Cloudflare or external traffic. Only authenticated requests through Open WebUI (`127.0.0.1:8080`) can trigger model generation.
3. **Authentication & RBAC**: Open WebUI enforces user login and session tokens. The initial registered account assumes primary administrator privileges.
4. **Research Isolation Boundary**: The Heretic research environment operates in an isolated virtual environment (`/Users/Sergiu/AI/research/heretic/venv`) and sandbox directory. Experimental weight manipulations are kept separate from production models.

---

## 2. Directory Layout

| Directory | Purpose |
|---|---|
| `/Users/Sergiu/AI/bin` | Operational lifecycle scripts (`start.sh`, `stop.sh`, `status.sh`, `diagnostics.sh`, `backup.sh`, `restore.sh`) |
| `/Users/Sergiu/AI/config` | Cloudflare tunnel tokens, ingress YAML, and Ollama environment flags |
| `/Users/Sergiu/AI/models` | Local model weights, manifests, and blobs (`OLLAMA_MODELS`) |
| `/Users/Sergiu/AI/data/open-webui` | Persistent chat history, user settings, and SQLite database (`webui.db`) |
| `/Users/Sergiu/AI/data/backups` | Timestamped `.tar.gz` recovery archives (auto-retaining last 7 snapshots) |
| `/Users/Sergiu/AI/logs` | Dedicated service logs (`ollama.log`, `open-webui.log`, `cloudflared.log`) |
| `/Users/Sergiu/AI/research/heretic` | Mathematical refusal ablation framework & Optuna parameter exploration |
| `/Users/Sergiu/AI/research/eval` | Benchmarking scripts (`run_benchmark.py`) and prompt evaluation datasets |

---

## 3. Operational Runbook

### Service Management
```bash
# Start all AI services (Ollama + Open WebUI + Cloudflare Tunnel)
/Users/Sergiu/AI/bin/start.sh

# Check real-time service status and endpoints
/Users/Sergiu/AI/bin/status.sh

# Run comprehensive system and GPU diagnostics
/Users/Sergiu/AI/bin/diagnostics.sh

# Stop all AI services
/Users/Sergiu/AI/bin/stop.sh
```

### Backup & Disaster Recovery
```bash
# Generate immediate timestamped snapshot
/Users/Sergiu/AI/bin/backup.sh

# Restore from a backup archive
/Users/Sergiu/AI/bin/restore.sh /Users/Sergiu/AI/data/backups/ai_backup_YYYYMMDD_HHMMSS.tar.gz
```

---

## 4. Heretic Research Tool & AI Safety Findings

### Methodology
Heretic investigates the mechanistic representation of safety alignment in decoder-only language models. Specifically:
- **Refusal Direction**: Identifies the primary linear subspace mediating refusals by computing the mean activation difference:
  $$\vec{v}_{refusal} = \frac{\mu_{sensitive} - \mu_{baseline}}{\|\mu_{sensitive} - \mu_{baseline}\|_2}$$
- **Orthogonal Weight Ablation**: Projects out the refusal direction from targeted feed-forward and attention projection layers:
  $$W' = W - \vec{v}_{refusal} \cdot (\vec{v}_{refusal}^T W)$$
- **Safety Trade-offs**: While directional ablation can reduce false refusals on benign technical queries, it also circumvents intended safety alignment. In our framework, these experiments are strictly confined to the research sandbox with adversarial negative controls and KL-divergence audits.
