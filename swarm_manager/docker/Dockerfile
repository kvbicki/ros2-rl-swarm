FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

COPY src/rl_agent.py ./rl_agent.py
COPY src/swarm_env.py ./swarm_env.py

CMD ["bash"]
