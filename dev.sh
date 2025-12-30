#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if [[ -f frontend/.env.example && ! -f frontend/.env ]]; then
  cp frontend/.env.example frontend/.env
  echo "Created frontend/.env from frontend/.env.example"
fi

python_bin="${PYTHON:-python3}"
if [[ ! -d .venv ]]; then
  "$python_bin" -m venv .venv
fi

venv_python="$repo_root/.venv/bin/python"
"$venv_python" -m pip install --upgrade pip
"$venv_python" -m pip install -r backend/requirements.txt

if [[ -d frontend/node_modules ]]; then
  npm -C frontend install
else
  npm -C frontend ci
fi

echo "Starting backend (http://127.0.0.1:8000)..."
"$venv_python" -m uvicorn app.main:app --app-dir backend --reload &
backend_pid="$!"
trap 'kill "$backend_pid" 2>/dev/null || true' EXIT

echo "Starting frontend (http://127.0.0.1:5173)..."
npm -C frontend run dev
