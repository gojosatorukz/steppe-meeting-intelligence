# Local-only stack: FastAPI + faster-whisper + sherpa-onnx + Ollama. No app/.
.PHONY: help preflight dev models frontend test health

help:
	@echo "make dev        run the API (uvicorn, port 8000)"
	@echo "make models     fetch diarization + VAD models, print remaining setup steps"
	@echo "make frontend   install deps and build the React dashboard into frontend/dist"
	@echo "make test       run the Python test suite"
	@echo "make preflight  check toolchain, GPU, models, Ollama, ports, mic"
	@echo "make health     curl /health and pretty-print it"

preflight:
	@./scripts/preflight.sh

dev:
	cd api && uv run uvicorn main:app --port 8000

models:
	@./api/scripts/get_diarization_models.sh
	@./api/scripts/get_vad_model.sh
	@echo
	@echo "Remaining manual steps:"
	@echo "  cd api && uv run ct2-transformers-converter --model shyngys879/kazakh-whisper-large-v3-turbo \\"
	@echo "    --output_dir models/kk-turbo-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization int8"
	@echo "  ollama pull qwen3.5:4b"

frontend:
	cd frontend && npm ci --no-audit --no-fund && npm run build

test:
	cd api && uv run pytest -q

health:
	@curl -s localhost:8000/health | python3 -m json.tool
