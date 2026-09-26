\
# vLLM Qwen3.8 hosting — convenience targets.
#
# Two separate paths:
#   1. Manual (serve/serve-nospec/stop/status/logs/bench...): start ONE
#      model directly, foreground or background, for local dev/benchmarking.
#      All these targets take MODEL=<name>, matching a directory under
#      models/ (default: qwen3.8-27b).
#   2. llama-swap (swap-*): the dynamic gateway that cold-starts whichever
#      model is requested and picks a free GPU for it automatically (see
#      pick_gpu_and_launch.sh) — this is the main way models get started
#      going forward; reach for the manual path only when you specifically
#      want to run one model in isolation (benchmarking, debugging a
#      checkpoint change, etc.) rather than through the shared gateway.
#
# Common usage:
#   make serve          start the spec-decode (dflash) server in the background
#   make status         check whether it's up, and which variant is running
#   make logs           tail the server log
#   make stop           stop whichever server variant is running
#   make restart        stop + serve
#   make swap-status    check llama-swap + which models are currently loaded
#   make swap-call MODEL=<name>   send a test request through the gateway
#
SHELL      := /bin/bash
MODEL      ?= qwen3.8-27b
MODEL_DIR  := models/$(MODEL)
ENV_FILE   := $(MODEL_DIR)/model.env
PROMPT     ?= hi

LLAMA_SWAP_SERVICE := bfg-llama-swap.service
LLAMA_SWAP_CONFIG   := llama-swap-config.yaml
LLAMA_SWAP_URL       := http://127.0.0.1:12000

.PHONY: help serve serve-fg serve-nospec serve-nospec-fg stop status logs logs-nospec restart bench bench-concurrency \
        swap-validate swap-status swap-restart swap-logs swap-call

help:
	@echo "Manual path (all take MODEL=<name>, default $(MODEL); dirs under models/):"
	@echo "  make serve              start spec-decode (dflash) server in background -> \$$MODEL_DIR/server.log"
	@echo "  make serve-fg           same, but in the foreground (Ctrl-C to stop)"
	@echo "  make serve-nospec       start baseline server (no speculative decoding) in background"
	@echo "  make serve-nospec-fg    same, but in the foreground"
	@echo "  make stop               stop whichever server variant is currently running"
	@echo "  make restart            stop, then start the spec-decode server"
	@echo "  make status             health check + report which variant is running"
	@echo "  make logs               tail -f \$$MODEL_DIR/server.log"
	@echo "  make logs-nospec        tail -f \$$MODEL_DIR/server_nospec.log"
	@echo "  make bench              run the prefill/decode depth sweep (spec vs nospec, ~10-20 min)"
	@echo "  make bench-concurrency  run the concurrency (1/2/4) sweep (spec vs nospec)"
	@echo ""
	@echo "llama-swap path (the main way models get started going forward):"
	@echo "  make swap-status        systemd service status + currently-loaded models"
	@echo "  make swap-validate      check $(LLAMA_SWAP_CONFIG) parses and llama-swap accepts it, without restarting"
	@echo "  make swap-restart       validate, then restart the service to pick up config changes"
	@echo "                          (WARNING: interrupts any currently-loaded model; prompts for confirmation)"
	@echo "  make swap-logs          tail -f the service's journal"
	@echo "  make swap-call MODEL=<name> [PROMPT=\"...\"]   send a test chat request through the gateway"
	@echo "                          (cold-starts MODEL if not already loaded — can take several minutes)"

serve:
	@source $(ENV_FILE); \
	if curl -s -o /dev/null "http://127.0.0.1:$$PORT/health"; then \
		echo "A server is already running on port $$PORT. Run 'make stop' first, or 'make restart'."; \
		exit 1; \
	fi; \
	nohup ./$(MODEL_DIR)/serve.sh > $(MODEL_DIR)/server.log 2>&1 & \
	echo "Spec-decode server ($(MODEL)) launching in background (PID $$!). Logs: $(MODEL_DIR)/server.log"; \
	echo "Check readiness with: make status MODEL=$(MODEL)"

serve-fg:
	./$(MODEL_DIR)/serve.sh

serve-nospec:
	@source $(ENV_FILE); \
	if curl -s -o /dev/null "http://127.0.0.1:$$PORT/health"; then \
		echo "A server is already running on port $$PORT. Run 'make stop' first, or 'make restart'."; \
		exit 1; \
	fi; \
	nohup ./$(MODEL_DIR)/serve_nospec.sh > $(MODEL_DIR)/server_nospec.log 2>&1 & \
	echo "No-spec baseline server ($(MODEL)) launching in background (PID $$!). Logs: $(MODEL_DIR)/server_nospec.log"; \
	echo "Check readiness with: make status MODEL=$(MODEL)"

serve-nospec-fg:
	./$(MODEL_DIR)/serve_nospec.sh

stop:
	@echo "Stopping any vLLM server for $(MODEL)..."
	-pkill -f "serve_nospec\.sh" 2>/dev/null
	-pkill -f "serve\.sh" 2>/dev/null
	-pkill -f "vllm serve" 2>/dev/null
	-pkill -f "VLLM::EngineCore" 2>/dev/null
	@source $(ENV_FILE); \
	echo "Waiting for port $$PORT to free..."; \
	waited=0; \
	while (echo > /dev/tcp/127.0.0.1/$$PORT) 2>/dev/null; do \
		sleep 2; waited=$$((waited + 2)); \
		if [ $$waited -ge 60 ]; then echo "Timed out waiting for port to free."; break; fi; \
	done
	@echo "Stopped."

restart: stop serve

status:
	@source $(ENV_FILE); \
	if curl -s -o /dev/null "http://127.0.0.1:$$PORT/health"; then \
		echo "Server is UP on port $$PORT ($(MODEL))."; \
		if pgrep -f "speculative-config" > /dev/null 2>&1 || ps aux | grep -q "[s]peculative_config"; then \
			echo "Variant: spec-decode (dflash)"; \
		else \
			echo "Variant: no-spec baseline (or variant undetermined — check ps aux)"; \
		fi; \
		curl -s "http://127.0.0.1:$$PORT/health" -o /dev/null -w "Health check HTTP status: %{http_code}\n"; \
	else \
		echo "No server responding on port $$PORT ($(MODEL))."; \
	fi

logs:
	tail -f $(MODEL_DIR)/server.log

logs-nospec:
	tail -f $(MODEL_DIR)/server_nospec.log

bench:
	./run_bench_sweep.sh $(MODEL)

bench-concurrency:
	./run_bench_concurrency.sh $(MODEL)

# --- llama-swap path ---

swap-validate:
	@echo "Checking $(LLAMA_SWAP_CONFIG) parses..."
	@python3 -c "import yaml; yaml.safe_load(open('$(LLAMA_SWAP_CONFIG)'))" && echo "YAML syntax OK"
	@echo "Checking llama-swap itself accepts it (dry run on a scratch port)..."
	@rm -f /tmp/llama-swap-validate.log; \
	timeout 5 ./llama-swap -config $(LLAMA_SWAP_CONFIG) -listen ":29347" > /tmp/llama-swap-validate.log 2>&1 & \
	SWAPPID=$$!; \
	sleep 2; \
	if grep -q "listening on" /tmp/llama-swap-validate.log; then \
		echo "llama-swap accepted the config"; \
	else \
		echo "llama-swap REJECTED the config:"; cat /tmp/llama-swap-validate.log; \
		kill $$SWAPPID 2>/dev/null; exit 1; \
	fi; \
	kill $$SWAPPID 2>/dev/null; true

swap-status:
	@echo "=== $(LLAMA_SWAP_SERVICE) ==="
	@systemctl --user status $(LLAMA_SWAP_SERVICE) --no-pager 2>&1 | head -6
	@echo ""
	@echo "=== currently loaded models ($(LLAMA_SWAP_URL)/running) ==="
	@curl -s $(LLAMA_SWAP_URL)/running | python3 -m json.tool 2>/dev/null \
		|| echo "gateway not reachable on $(LLAMA_SWAP_URL) — is $(LLAMA_SWAP_SERVICE) running?"

# Depends on swap-validate so a bad config fails loudly here instead of after
# the currently-running model has already been killed by the restart.
swap-restart: swap-validate
	@echo ""
	@RUNNING=$$(curl -s $(LLAMA_SWAP_URL)/running | python3 -c "import json,sys; d=json.load(sys.stdin); m=[x['model'] for x in d.get('running',[])]; print(', '.join(m) if m else 'none')" 2>/dev/null || echo "unknown"); \
	echo "Currently loaded: $$RUNNING"; \
	echo "Restarting $(LLAMA_SWAP_SERVICE) will interrupt any of those (in-flight"; \
	echo "requests fail; next request cold-starts again, ~2-5 min)."; \
	read -p "Proceed? [y/N] " ans; \
	if [ "$$ans" != "y" ] && [ "$$ans" != "Y" ]; then echo "Aborted."; exit 1; fi; \
	systemctl --user restart $(LLAMA_SWAP_SERVICE)
	@sleep 2
	@systemctl --user status $(LLAMA_SWAP_SERVICE) --no-pager 2>&1 | head -6

swap-logs:
	journalctl --user -u $(LLAMA_SWAP_SERVICE) -f

# Builds the JSON body with python (not raw shell interpolation) so PROMPT
# can contain quotes/special characters without breaking the request.
swap-call:
	@BODY=$$(python3 -c "import json,sys; print(json.dumps({'model': sys.argv[1], 'messages': [{'role':'user','content': sys.argv[2]}], 'max_tokens': 20}))" "$(MODEL)" "$(PROMPT)"); \
	echo "POST $(LLAMA_SWAP_URL)/v1/chat/completions (model=$(MODEL)) — cold-starts it if not already loaded, can take several minutes..."; \
	curl -s $(LLAMA_SWAP_URL)/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d "$$BODY" \
	  -w "\nHTTP status: %{http_code}\ntotal time: %{time_total}s\n" \
	  --max-time 600
