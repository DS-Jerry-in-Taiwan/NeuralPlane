#!/usr/bin/env bash
# ==============================================================================
# NeuralPlane — 環境變數範本
#
# 使用方式：
#   cp env.example.sh env.sh
#   vim env.sh          # 填入你的 secret / ID
#   source env.sh       # 載入環境變數
#
# ⚠️  不要 commit env.sh（內含 API key）。
#     這個範本檔 (env.example.sh) 不含 secret，可以 commit。
# ==============================================================================

# ---------------------------------------------------------------------------
# 1. LLM Provider（Phase 1C — OpenAI 相容）
# ---------------------------------------------------------------------------
#
# 要使用真實 LLM API，執行時加 --provider openai。
# 使用 fake/static provider 則不需要設這些（直接用 --provider static）。
#
export LLM_API_KEY=""                # OpenAI API key（或 LiteLLM proxy 的 key）
export LLM_MODEL="gpt-4o"           # 模型名稱（litellm proxy 則設 proxy 上的 model id）
export LLM_BASE_URL="https://api.openai.com/v1"   # API base URL
                                                   # LiteLLM 時改為 http://localhost:4000/v1
export LLM_TEMPERATURE="0.3"         # 取樣溫度 (0.0 ~ 2.0，建議 0.3 以下)
export LLM_MAX_TOKENS="4096"         # 每次回應最大 token 數

# ---------------------------------------------------------------------------
# 2. Plane API（Phase 2 以後 — live write 時需要）
# ---------------------------------------------------------------------------
#
# 目前 Phase 1 全部走 dry-run／mock，不需要設這些。
# 等進入 Phase 2（live Plane integration）才需要填。
#
export PLANE_API_KEY=""              # Plane API token
export PLANE_BASE_URL="https://api.plane.so"       # Plane API base URL
export PLANE_WORKSPACE_SLUG=""       # 你的 workspace 名稱
export PLANE_PROJECT_ID=""           # 目標 project UUID
export PLANE_DEFAULT_STATE_ID=""     # 新 issue 的預設 state UUID

# ---------------------------------------------------------------------------
# 3. LiteLLM Proxy（選擇性 — 開發機用）
# ---------------------------------------------------------------------------
#
# 如果使用 litellm proxy 取代直連 OpenAI，把 LLM_BASE_URL 改為：
#
#   export LLM_BASE_URL="http://localhost:4000/v1"
#
# Proxy 啟動方式（需先 pip install litellm）：
#
#   litellm --model claude-3-opus-20240229 --port 4000
#   litellm --model gpt-4o --model claude-3-opus --fallback gpt-3.5-turbo --port 4000
#
# ---------------------------------------------------------------------------
# 4. Smoke test（確認環境變數已載入）
# ---------------------------------------------------------------------------
#
#   echo "$LLM_API_KEY"        # 應顯示你設的值
#   echo "$PLANE_BASE_URL"     # 應顯示 https://api.plane.so
#
# 所有值確認後執行：
#
#   python -m src.neuralplane.cli_e2e_dry_run \
#       --input examples/task_request.sample.txt \
#       --provider openai
