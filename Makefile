# qwen3.8-flash-next hosting — convenience targets.
#
# The model runs as an always-on systemd user service (deploy/bfg-flash-next.service)
# serving vLLM directly on port 8000. `serve-fg`
# runs it in the foreground for debugging (stop the service first — same port, same GPU).
#
#   make install        render the unit into ~/.config/systemd/user, daemon-reload
#   make start|stop|restart|status|logs
#   make call [PROMPT=...]   send a test chat request
#   make bench / bench-concurrency   sweeps against the running server
#
SHELL    := /bin/bash
MODEL    := qwen3.8-flash-next
SERVICE  := bfg-flash-next.service
UNIT_DIR := $(HOME)/.config/systemd/user
PROMPT   ?= hi
PORT     := $(shell sed -n 's/^PORT=//p' models/$(MODEL)/model.env)
URL      := http://127.0.0.1:$(PORT)

.PHONY: help docker-build docker-run install start stop restart status logs call serve-fg bench bench-concurrency

help:
	@echo "  make install            render deploy/$(SERVICE) into $(UNIT_DIR)"
	@echo "  make start|stop|restart control the service (restart/stop interrupt requests; cold start ~3-4 min)"
	@echo "  make status             service state + /health + loaded model"
	@echo "  make logs               tail -f the service journal"
	@echo "  make call [PROMPT=...]  send a test chat request to $(URL)"
	@echo "  make serve-fg           run serve.sh in the foreground (stop the service first)"
	@echo "  make docker-build|docker-run   image build / run on DOCKER_GPU=$(DOCKER_GPU) at :$(DOCKER_PORT)"
	@echo "  make bench              prefill/decode depth sweep against the running server"
	@echo "  make bench-concurrency  concurrency (2/4) sweep against the running server"

# --- Docker (see docs/docker-image.md) ---
IMAGE      ?= qwen3.8-flash-next:vllm0.32
DOCKER_GPU ?= 1
DOCKER_PORT ?= 8002
# Host dir holding the weights, mounted read-only at the SAME path in the container.
# Set it (and MODEL_PATH, which must lie under it) to serve weights from elsewhere.
DOCKER_MODELS ?= /data/huggingface
# One persistent dir for every cache that otherwise lives in the container's
# throwaway layer: FlashInfer JIT kernels (HOME/.cache/flashinfer), vLLM's
# torch.compile + autotune caches (VLLM_CACHE_ROOT), triton. Runs as the host
# user so nothing in it ends up root-owned.
DOCKER_CACHE ?= $(HOME)/.cache/flash-next-docker

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	mkdir -p $(DOCKER_CACHE)
	docker run --rm --name flash-next-gpu$(DOCKER_GPU) \
	  --user $$(id -u):$$(id -g) \
	  --device nvidia.com/gpu=$(DOCKER_GPU) --ipc=host --ulimit memlock=-1:-1 \
	  -v $(DOCKER_MODELS):$(DOCKER_MODELS):ro \
	  $(if $(MODEL_PATH),-e MODEL_PATH=$(MODEL_PATH)) \
	  -v $(DOCKER_CACHE):/cache \
	  -e HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton \
	  -p $(DOCKER_PORT):8000 $(IMAGE)

install:
	mkdir -p $(UNIT_DIR)
	sed 's|@REPO@|$(CURDIR)|g' deploy/$(SERVICE) > $(UNIT_DIR)/$(SERVICE)
	systemctl --user daemon-reload
	@echo "Installed. 'make start' to run it; 'systemctl --user enable $(SERVICE)' to start on login/boot."

start:
	systemctl --user start $(SERVICE)

stop:
	systemctl --user stop $(SERVICE)

restart:
	systemctl --user restart $(SERVICE)

status:
	@systemctl --user status $(SERVICE) --no-pager 2>&1 | head -6
	@echo ""
	@if curl -s -o /dev/null -w "health: HTTP %{http_code}\n" $(URL)/health; then \
		curl -s $(URL)/v1/models | python3 -c "import json,sys; print('serving:', [m['id'] for m in json.load(sys.stdin)['data']])"; \
	else echo "no server responding on $(URL)"; fi

logs:
	journalctl --user -u $(SERVICE) -f

# Builds the JSON body with python so PROMPT can contain quotes/special characters.
call:
	@BODY=$$(python3 -c "import json,sys; print(json.dumps({'model': sys.argv[1], 'messages': [{'role':'user','content': sys.argv[2]}], 'max_tokens': 20}))" "$(MODEL)" "$(PROMPT)"); \
	curl -s $(URL)/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d "$$BODY" \
	  -w "\nHTTP status: %{http_code}\ntotal time: %{time_total}s\n" \
	  --max-time 600

serve-fg:
	./models/$(MODEL)/serve.sh

bench:
	./run_bench.sh depth

bench-concurrency:
	./run_bench.sh concurrency
