# Vendored llama.cpp is pinned to the commit llama-cpp-python 0.3.36 ships.
# Convert, quantize and eval all use it so the GGUF format never drifts.
LLCP_VER := 0.3.36
LLAMA_CPP := vendor/sdist/llama_cpp_python-$(LLCP_VER)/vendor/llama.cpp
SDK := $(shell xcrun --show-sdk-path)
# Stale CLT headers shadow the SDK's libc++ on this machine. Point clang at the SDK copy.
CXX_FIX := -nostdinc++ -isystem $(SDK)/usr/include/c++/v1

.PHONY: setup llama-cpp model smoke test

setup:
	CXXFLAGS="$(CXX_FIX)" CMAKE_ARGS="-DGGML_METAL=on" uv sync

llama-cpp:
	mkdir -p vendor/sdist
	cd vendor/sdist && uv run python -c "import json,urllib.request as u;d=json.load(u.urlopen('https://pypi.org/pypi/llama-cpp-python/$(LLCP_VER)/json'));u.urlretrieve([x['url'] for x in d['urls'] if x['packagetype']=='sdist'][0],'llcp.tar.gz')" && tar xzf llcp.tar.gz
	cmake -S $(LLAMA_CPP) -B $(LLAMA_CPP)/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CXX_FLAGS="$(CXX_FIX)"
	cmake --build $(LLAMA_CPP)/build --target llama-quantize -j4

model:
	uv run hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir models/qwen2.5-coder-1.5b-instruct

smoke:
	uv run python scripts/smoke_hf.py
	uv run python scripts/smoke_lora.py
	uv run python scripts/smoke_gguf.py

test:
	uv run pytest -q
