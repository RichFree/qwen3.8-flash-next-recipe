\
# vLLM Qwen3.8 hosting — convenience targets.
#
# Common usage:
#   make serve          start the spec-decode (dflash) server in the background
#   make status         check whether it's up, and which variant is running
#   make logs           tail the server log
#   make stop           stop whichever server variant is running
#   make restart        stop + serve
#
# All targets take MODEL=<name>, matching a directory under models/
# (e.g. MODEL=qwen3.8-27b, the default; or MODEL=qwen3.8-flash-next — see
# that model's README.md before serving it, it's an unvalidated scaffold).
#
SHELL      := /bin/bash
MODEL      ?= qwen3.8-27b
MODEL_DIR  := models/$(MODEL)
ENV_FILE   := $(MODEL_DIR)/model.env

.PHONY: help serve serve-fg serve-nospec serve-nospec-fg stop status logs logs-nospec restart bench bench-concurrency

help:
	@echo "Targets (all take MODEL=<name>, default $(MODEL); dirs under models/):"
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
