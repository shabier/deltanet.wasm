#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <emscripten/bind.h>

static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static llama_sampler * g_smpl = nullptr;
static const llama_vocab * g_vocab = nullptr;

bool loadModel(const std::string & path, int n_ctx) {
    ggml_backend_load_all();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_model) return false;

    g_vocab = llama_model_get_vocab(g_model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 2048;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.no_perf = true;
    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        return false;
    }

    auto sparams = llama_sampler_chain_default_params();
    g_smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_smpl, llama_sampler_init_greedy());
    return true;
}

int tokenize(const std::string & text) {
    if (!g_vocab) return -1;
    const int n = -llama_tokenize(g_vocab, text.c_str(), text.size(), NULL, 0, true, true);
    return n;
}

bool decodePrompt(const std::string & text) {
    if (!g_model || !g_ctx || !g_vocab) return false;

    const int n = -llama_tokenize(g_vocab, text.c_str(), text.size(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(g_vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);

    // Process in batches to stay within n_batch limit
    const int n_batch = 2048;
    for (int i = 0; i < n; i += n_batch) {
        int batch_size = std::min(n_batch, n - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, batch_size);
        if (llama_decode(g_ctx, batch) != 0) return false;
    }
    return true;
}

std::string generateToken() {
    if (!g_model || !g_ctx || !g_vocab || !g_smpl) return "";

    llama_token tok = llama_sampler_sample(g_smpl, g_ctx, -1);
    if (llama_vocab_is_eog(g_vocab, tok)) return "";

    char buf[256];
    int len = llama_token_to_piece(g_vocab, tok, buf, sizeof(buf), 0, true);
    if (len <= 0) return "";

    llama_batch batch = llama_batch_get_one(&tok, 1);
    if (llama_decode(g_ctx, batch) != 0) return "";

    return std::string(buf, len);
}

void resetContext() {
    if (g_ctx) llama_memory_clear(llama_get_memory(g_ctx), true);
}

void freeModel() {
    if (g_smpl) { llama_sampler_free(g_smpl); g_smpl = nullptr; }
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_vocab = nullptr;
}

EMSCRIPTEN_BINDINGS(deltanet_wasm) {
    emscripten::function("loadModel", &loadModel);
    emscripten::function("tokenize", &tokenize);
    emscripten::function("decodePrompt", &decodePrompt);
    emscripten::function("generateToken", &generateToken);
    emscripten::function("resetContext", &resetContext);
    emscripten::function("freeModel", &freeModel);
}
