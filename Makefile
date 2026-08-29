\
# vLLM Qwen3.8-27B hosting — convenience targets.
#
# Common usage:
#   make serve          start the spec-decode (dflash) server in the background
#   make status         check whether it's up, and which variant is running
#   make logs           tail the server log
#   make stop           stop whichever server variant is running
#   make restart        stop + serve
#
PORT       := 12000
HEALTH_URL := http://127.0.0.1:$(PORT)/health
SHELL      := /bin/bash

.PHONY: help serve serve-fg serve-nospec serve-nospec-fg stop status logs logs-nospec restart bench bench-concurrency

help:
	@echo "Targets:"
	@echo "  make serve              start spec-decode (dflash) server in background -> server.log"
	@echo "  make serve-fg           same, but in the foreground (Ctrl-C to stop)"
	@echo "  make serve-nospec       start baseline server (no speculative decoding) in background -> server_nospec.log"
	@echo "  make serve-nospec-fg    same, but in the foreground"
	@echo "  make stop               stop whichever server variant is currently running"
	@echo "  make restart            stop, then start the spec-decode server"
	@echo "  make status             health check + report which variant is running"
	@echo "  make logs               tail -f server.log"
	@echo "  make logs-nospec        tail -f server_nospec.log"
	@echo "  make bench              run the prefill/decode depth sweep (spec vs nospec, ~10-20 min)"
	@echo "  make bench-concurrency  run the concurrency (1/2/4) sweep (spec vs nospec)"

serve:
	@if curl -s -o /dev/null "$(HEALTH_URL)"; then \
		echo "A server is already running on port $(PORT). Run 'make stop' first, or 'make restart'."; \
		exit 1; \
	fi
	nohup ./serve.sh > server.log 2>&1 &
	@echo "Spec-decode server launching in background (PID $$!). Logs: server.log"
	@echo "Check readiness with: make status"

serve-fg:
	./serve.sh

serve-nospec:
	@if curl -s -o /dev/null "$(HEALTH_URL)"; then \
		echo "A server is already running on port $(PORT). Run 'make stop' first, or 'make restart'."; \
		exit 1; \
	fi
	nohup ./serve_nospec.sh > server_nospec.log 2>&1 &
	@echo "No-spec baseline server launching in background (PID $$!). Logs: server_nospec.log"
	@echo "Check readiness with: make status"

serve-nospec-fg:
	./serve_nospec.sh

stop:
	@echo "Stopping any vLLM server on port $(PORT)..."
	-pkill -f "serve_nospec.sh" 2>/dev/null
	-pkill -f "\./serve\.sh" 2>/dev/null
	-pkill -f "vllm serve" 2>/dev/null
	-pkill -f "VLLM::EngineCore" 2>/dev/null
	@echo "Waiting for port $(PORT) to free..."
	@waited=0; \
	while (echo > /dev/tcp/127.0.0.1/$(PORT)) 2>/dev/null; do \
		sleep 2; waited=$$((waited + 2)); \
		if [ $$waited -ge 60 ]; then echo "Timed out waiting for port to free."; break; fi; \
	done
	@echo "Stopped."

restart: stop serve

status:
	@if curl -s -o /dev/null "$(HEALTH_URL)"; then \
		echo "Server is UP on port $(PORT)."; \
		if pgrep -f "speculative-config" > /dev/null 2>&1 || ps aux | grep -q "[s]peculative_config"; then \
			echo "Variant: spec-decode (dflash)"; \
		else \
			echo "Variant: no-spec baseline (or variant undetermined — check ps aux)"; \
		fi; \
		curl -s "$(HEALTH_URL)" -o /dev/null -w "Health check HTTP status: %{http_code}\n"; \
	else \
		echo "No server responding on port $(PORT)."; \
	fi

logs:
	tail -f server.log

logs-nospec:
	tail -f server_nospec.log

bench:
	./run_bench_sweep.sh

bench-concurrency:
	./run_bench_concurrency.sh
