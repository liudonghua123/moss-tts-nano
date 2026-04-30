#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <fstream>
#include "onnxruntime_cxx_api.h"
#include "wav_writer.h"
#include "sentencepiece_processor.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace std;

#ifdef _WIN32
wstring ToOrtPath(const string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}
#else
string ToOrtPath(const string& str) { return str; }
#endif

struct Args {
    string model_dir = "onnx/MOSS-TTS-Nano-100M-ONNX";
    string codec_dir = "onnx/MOSS-Audio-Tokenizer-Nano-ONNX";
    string voice_dir = "voices";
    string output = "output.wav";
    string text = "";
    string voice_name = "junhao";  // Default voice name
    vector<int32_t> ids;
    bool verbose = false;
    int max_new_frames = 100;
};

// Codec config from codec_browser_onnx_meta.json
const int CODEC_SAMPLE_RATE = 48000;
const int CODEC_CHANNELS = 2;
const int CODEC_NUM_QUANTIZERS = 16;
const int64_t N_VQ = 16;
const int64_t ROW_WIDTH = 17;
const int64_t HIDDEN_DIM = 768;
const int64_t VOCAB_SIZE = 16384;
const int64_t CODEBOOK_SIZE = 1024;
const int32_t AUDIO_PAD_TOKEN = 1024;
const int32_t AUDIO_ASSISTANT_SLOT_TOKEN = 9;
const int32_t AUDIO_END_TOKEN = 7;
const int32_t AUDIO_USER_SLOT_TOKEN = 8;
const int32_t AUDIO_START_TOKEN = 6;
const int NUM_GLOBAL_LAYERS = 12;
const int NUM_GLOBAL_HEADS = 12;
const int HEAD_DIM = 64;

// Prompt template token IDs from browser_poc_manifest.json
const vector<int32_t> USER_PROMPT_PREFIX = {600, 289, 10356, 13, 10356, 10425, 1860, 4546, 12907, 10363, 13325, 11492};
const vector<int32_t> USER_PROMPT_AFTER_REF = {
    10356, 10425, 3965, 7738, 11492, 505, 587,
    10356, 10425, 352, 500, 856, 11492, 505, 587,
    10356, 10425, 2128, 1247, 594, 11492, 505, 587,
    10356, 10425, 348, 909, 561, 3648, 11492, 505, 587,
    10356, 10425, 2818, 1305, 355, 348, 909, 11492, 505, 587,
    10356, 10425, 484, 339, 10367, 783, 11492, 505, 587,
    10356, 10425, 2427, 980, 11492
};
const vector<int32_t> ASSISTANT_PROMPT_PREFIX = {10356, 14, 5, 4, 8165, 430};

// Built-in voice: Junhao (Chinese Male) - prompt_audio_codes from browser_poc_manifest.json
const vector<vector<int32_t>> JUNHAO_PROMPT_CODES = {
    {922, 507, 253, 140, 700, 228, 523, 612, 20, 194, 877, 682, 415, 244, 181, 195},
    {503, 720, 733, 615, 153, 759, 1023, 541, 744, 1015, 245, 68, 323, 183, 307, 854},
    {829, 958, 621, 750, 839, 865, 683, 955, 102, 557, 317, 968, 391, 539, 671, 425},
    {295, 776, 279, 19, 61, 173, 65, 189, 923, 586, 42, 125, 442, 341, 462, 90},
    {556, 787, 992, 553, 436, 483, 618, 160, 504, 550, 402, 169, 607, 576, 72, 507},
    {537, 11, 699, 752, 135, 176, 122, 587, 722, 1016, 732, 103, 818, 680, 794, 605},
    {314, 426, 584, 58, 853, 859, 349, 113, 295, 128, 939, 692, 225, 203, 107, 881},
    {568, 471, 263, 796, 587, 485, 874, 63, 918, 571, 191, 528, 308, 807, 979, 412},
    {363, 1018, 425, 985, 140, 7, 264, 833, 943, 111, 568, 241, 630, 759, 444, 164},
    {101, 956, 551, 11, 250, 581, 458, 487, 105, 74, 2, 382, 141, 818, 414, 652},
};

// Helper function: build text rows
vector<vector<int32_t>> buildTextRows(const vector<int32_t>& token_ids) {
    vector<vector<int32_t>> rows;
    for (int32_t token_id : token_ids) {
        vector<int32_t> row(ROW_WIDTH, AUDIO_PAD_TOKEN);
        row[0] = token_id;
        rows.push_back(row);
    }
    return rows;
}

// Helper function: build audio prefix rows
vector<vector<int32_t>> buildAudioPrefixRows(const vector<vector<int32_t>>& prompt_audio_codes) {
    vector<vector<int32_t>> rows;
    for (const auto& code_row : prompt_audio_codes) {
        vector<int32_t> row(ROW_WIDTH, AUDIO_PAD_TOKEN);
        row[0] = AUDIO_USER_SLOT_TOKEN;
        for (int i = 0; i < N_VQ && i < (int)code_row.size(); ++i) {
            row[i + 1] = code_row[i];
        }
        rows.push_back(row);
    }
    return rows;
}

// Helper: build complete voice clone request rows
pair<vector<vector<int32_t>>, int> buildVoiceCloneRequestRows(
    const vector<vector<int32_t>>& prompt_audio_codes,
    const vector<int32_t>& text_token_ids) {

    // Build all token ID sequences
    vector<int32_t> prefix_text_ids = USER_PROMPT_PREFIX;
    prefix_text_ids.push_back(AUDIO_START_TOKEN);

    vector<int32_t> suffix_text_ids;
    suffix_text_ids.push_back(AUDIO_END_TOKEN);
    suffix_text_ids.insert(suffix_text_ids.end(), USER_PROMPT_AFTER_REF.begin(), USER_PROMPT_AFTER_REF.end());
    suffix_text_ids.insert(suffix_text_ids.end(), text_token_ids.begin(), text_token_ids.end());
    suffix_text_ids.insert(suffix_text_ids.end(), ASSISTANT_PROMPT_PREFIX.begin(), ASSISTANT_PROMPT_PREFIX.end());
    suffix_text_ids.push_back(AUDIO_START_TOKEN);

    // Build rows
    vector<vector<int32_t>> rows;
    rows = buildTextRows(prefix_text_ids);
    auto audio_rows = buildAudioPrefixRows(prompt_audio_codes);
    rows.insert(rows.end(), audio_rows.begin(), audio_rows.end());
    auto suffix_rows = buildTextRows(suffix_text_ids);
    rows.insert(rows.end(), suffix_rows.begin(), suffix_rows.end());

    return {rows, (int)rows.size()};
}

int main(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            cout << "Usage: moss-tts-nano [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --text <text>          Text to synthesize (required)" << endl;
            cout << "  --out <file>           Output WAV file (default: output.wav)" << endl;
            cout << "  --voice <name>         Voice name (default: junhao, or use audio file path)" << endl;
            cout << "  --voice-dir <dir>      Voice directory (default: voices)" << endl;
            cout << "  --model-dir <dir>      TTS model directory (default: onnx/MOSS-TTS-Nano-100M-ONNX)" << endl;
            cout << "  --codec-dir <dir>      Codec model directory (default: onnx/MOSS-Audio-Tokenizer-Nano-ONNX)" << endl;
            cout << "  --max-frames <n>       Max audio frames to generate (default: 100)" << endl;
            cout << "  --verbose              Print verbose output" << endl;
            cout << "Examples:" << endl;
            cout << "  moss-tts-nano --text \"Hello world\" --out hello.wav" << endl;
            cout << "  moss-tts-nano --text \"你好世界\" --voice /path/to/voice.wav" << endl;
            cout << "  moss-tts-nano --text \"你好\" --voice junhao  (from voices/ dir)" << endl;
            return 0;
        }
        else if (arg == "--models" || arg == "--model-dir") args.model_dir = argv[++i];
        else if (arg == "--codec-dir") args.codec_dir = argv[++i];
        else if (arg == "--voice-dir") args.voice_dir = argv[++i];
        else if (arg == "--out" || arg == "--output") args.output = argv[++i];
        else if (arg == "--text") args.text = argv[++i];
        else if (arg == "--voice") args.voice_name = argv[++i];
        else if (arg == "--verbose") args.verbose = true;
        else if (arg == "--max-frames") args.max_new_frames = stoi(argv[++i]);
        else if (arg == "--ids") {
            for (int j = i + 1; j < argc; ++j) {
                if (string(argv[j]).find("--") == 0) break;
                args.ids.push_back(static_cast<int32_t>(stoll(argv[j])));
                i = j;
            }
        }
    }

    if (!args.text.empty()) {
        sentencepiece::SentencePieceProcessor processor;
        string tokenizer_path = args.model_dir + "/tokenizer.model";
        if (!processor.Load(tokenizer_path).ok()) {
            // Try alternative path
            tokenizer_path = args.model_dir + "/sp.model";
            if (!processor.Load(tokenizer_path).ok()) {
                cerr << "Error: Cannot load tokenizer from " << args.model_dir << endl;
                cerr << "Please ensure tokenizer.model or sp.model exists in model directory." << endl;
                return 1;
            }
        }
        vector<int> sp_ids;
        processor.Encode(args.text, &sp_ids);
        for (int id : sp_ids) args.ids.push_back(static_cast<int32_t>(id));
    }

    if (args.ids.empty()) { cerr << "Error: No input (tokenizer failed or --ids not provided)." << endl; return 1; }
    if (args.verbose) cout << "Encoded " << args.ids.size() << " text tokens" << endl;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MOSS-TTS");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto start_total = chrono::high_resolution_clock::now();

        mt19937 rng(1234);

        // === 1. LOAD MODELS ===
        if (args.verbose) cout << "Loading models..." << endl;

        Ort::Session sess_prefill(env, ToOrtPath(args.model_dir + "/moss_tts_prefill.onnx").c_str(), opts);
        Ort::Session sess_decode(env, ToOrtPath(args.model_dir + "/moss_tts_decode_step.onnx").c_str(), opts);
        Ort::Session sess_local_frame(env, ToOrtPath(args.model_dir + "/moss_tts_local_fixed_sampled_frame.onnx").c_str(), opts);
        Ort::Session sess_codec_encode(env, ToOrtPath(args.codec_dir + "/moss_audio_tokenizer_encode.onnx").c_str(), opts);
        Ort::Session sess_codec_decode(env, ToOrtPath(args.codec_dir + "/moss_audio_tokenizer_decode_full.onnx").c_str(), opts);

        // Get decode output names
        vector<string> decode_out_names;
        for (size_t i = 0; i < sess_decode.GetOutputCount(); ++i) {
            auto n = sess_decode.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            decode_out_names.push_back(n.get());
        }
        vector<const char*> decode_out_ptrs;
        for (const auto& n : decode_out_names) decode_out_ptrs.push_back(n.c_str());

        // === 2. ENCODE PROMPT AUDIO (if provided) ===
        vector<vector<int32_t>> prompt_codes;

        // Check if it's a built-in voice name
        bool is_path = (args.voice_name.find('/') != string::npos ||
                       args.voice_name.find('\\') != string::npos ||
                       args.voice_name.find(".wav") != string::npos);

        if (!is_path && args.voice_name == "junhao") {
            // Use built-in Junhao voice
            prompt_codes = JUNHAO_PROMPT_CODES;
            if (args.verbose) cout << "Using built-in Junhao voice" << endl;
        } else {
            // Load from file (voice name maps to voice_dir/name.wav, or absolute path)
            string audio_path;
            if (is_path) {
                audio_path = args.voice_name;
            } else {
                audio_path = args.voice_dir + "/" + args.voice_name + ".wav";
            }

            if (args.verbose) cout << "Loading voice: " << audio_path << endl;

            // Load WAV file
            ifstream audio_file(audio_path, ios::binary);
            if (!audio_file) {
                cerr << "Error: Cannot open audio file: " << audio_path << endl;
                return 1;
            }

            // Read WAV header
            char riff[4], wave[4], fmt[4], data[4];
            audio_file.read(riff, 4);
            audio_file.read(wave, 4);
            audio_file.read(fmt, 4);
            audio_file.read(data, 4);

            uint32_t fmt_size, data_size;
            audio_file.read(reinterpret_cast<char*>(&fmt_size), 4);
            audio_file.read(reinterpret_cast<char*>(&data_size), 4);

            // Skip fmt chunk
            audio_file.seekg(fmt_size - 8, ios::cur);

            // Find data chunk
            while (audio_file.good()) {
                audio_file.read(data, 4);
                if (!audio_file.good()) break;
                audio_file.read(reinterpret_cast<char*>(&data_size), 4);
                if (string(data, 4) == "data") break;
                audio_file.seekg(data_size, ios::cur);
            }

            if (string(data, 4) != "data") {
                cerr << "Error: Cannot find data chunk in WAV file" << endl;
                return 1;
            }

            // Read audio samples
            size_t num_samples = data_size / 2;
            vector<int16_t> samples(num_samples);
            audio_file.read(reinterpret_cast<char*>(samples.data()), data_size);
            audio_file.close();

            if (args.verbose) cout << "Loaded " << num_samples << " samples from prompt audio" << endl;

            // Convert to float and reshape for encoder: [channels, samples]
            // Assume mono or stereo input, resample to 48000 if needed
            // For simplicity, we'll use the samples directly as mono at original rate
            // A full implementation would resample if needed

            // Create waveform tensor [1, channels, samples]
            vector<float> waveform(num_samples);
            for (size_t i = 0; i < num_samples; ++i) {
                waveform[i] = samples[i] / 32768.0f;
            }

            // Get encoder input names
            vector<string> enc_in_names;
            for (size_t i = 0; i < sess_codec_encode.GetInputCount(); ++i) {
                auto n = sess_codec_encode.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                enc_in_names.push_back(n.get());
            }
            vector<string> enc_out_names;
            for (size_t i = 0; i < sess_codec_encode.GetOutputCount(); ++i) {
                auto n = sess_codec_encode.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                enc_out_names.push_back(n.get());
            }

            // Prepare encoder inputs
            // Waveform shape: [1, num_samples] for encoder
            int64_t wave_shape[] = { 1, (int64_t)num_samples };
            Ort::Value wave_val = Ort::Value::CreateTensor<float>(
                mem_info, waveform.data(), waveform.size(), wave_shape, 2);

            int64_t len_shape[] = { 1 };
            int32_t wave_len = (int32_t)num_samples;
            Ort::Value len_val = Ort::Value::CreateTensor<int32_t>(
                mem_info, &wave_len, 1, len_shape, 1);

            vector<const char*> enc_in_ptrs;
            for (const auto& n : enc_in_names) enc_in_ptrs.push_back(n.c_str());
            vector<const char*> enc_out_ptrs;
            for (const auto& n : enc_out_names) enc_out_ptrs.push_back(n.c_str());
            Ort::Value enc_in_vals[] = { std::move(wave_val), std::move(len_val) };

            // Run encoder
            auto enc_results = sess_codec_encode.Run(
                Ort::RunOptions{nullptr},
                enc_in_ptrs.data(), enc_in_vals, 2,
                enc_out_ptrs.data(), enc_out_ptrs.size()
            );

            // Extract audio codes
            auto codes_shape = enc_results[0].GetTensorTypeAndShapeInfo().GetShape();
            int32_t* codes = enc_results[0].GetTensorMutableData<int32_t>();
            int32_t* code_len = enc_results[1].GetTensorMutableData<int32_t>();

            int64_t num_frames = code_len[0];
            if (args.verbose) cout << "Encoded prompt audio: " << num_frames << " frames" << endl;

            // Extract prompt codes
            prompt_codes.clear();
            for (int64_t i = 0; i < num_frames; ++i) {
                vector<int32_t> frame;
                for (int j = 0; j < CODEC_NUM_QUANTIZERS; ++j) {
                    frame.push_back(codes[i * CODEC_NUM_QUANTIZERS + j]);
                }
                prompt_codes.push_back(frame);
            }
        }

        // === 3. BUILD VOICE CLONE REQUEST ===
        if (args.verbose) cout << "Building voice clone request..." << endl;

        auto [request_rows, seq_len] = buildVoiceCloneRequestRows(prompt_codes, args.ids);
        if (args.verbose) {
            cout << "Request sequence length: " << seq_len << endl;
            cout << "Text tokens: " << args.ids.size() << endl;
            cout << "Prompt codes: " << prompt_codes.size() << " frames" << endl;
        }

        // === 3. PREFILL ===
        if (args.verbose) cout << "Running prefill..." << endl;

        // Flatten request_rows to input_ids tensor
        int64_t in_shape[] = { 1, seq_len, ROW_WIDTH };
        vector<int32_t> padded_ids(seq_len * ROW_WIDTH);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < ROW_WIDTH; ++j) {
                padded_ids[i * ROW_WIDTH + j] = request_rows[i][j];
            }
        }
        Ort::Value in_tens = Ort::Value::CreateTensor<int32_t>(mem_info, padded_ids.data(), padded_ids.size(), in_shape, 3);

        // Build attention_mask [1, seq_len] - all ones (attend to everything)
        int64_t mask_shape[] = { 1, seq_len };
        vector<int32_t> mask_data(seq_len, 1);
        Ort::Value mask_tens = Ort::Value::CreateTensor<int32_t>(mem_info, mask_data.data(), mask_data.size(), mask_shape, 2);

        // Run prefill with all outputs
        const char* prefill_in[] = { "input_ids", "attention_mask" };
        vector<string> prefill_out_names;
        for (size_t i = 0; i < sess_prefill.GetOutputCount(); ++i) {
            auto n = sess_prefill.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            prefill_out_names.push_back(n.get());
        }
        vector<const char*> prefill_out_ptrs;
        for (const auto& n : prefill_out_names) prefill_out_ptrs.push_back(n.c_str());

        Ort::Value prefill_in_vals[] = { std::move(in_tens), std::move(mask_tens) };

        auto prefill_results = sess_prefill.Run(
            Ort::RunOptions{nullptr},
            prefill_in, prefill_in_vals, 2,
            prefill_out_ptrs.data(), prefill_out_ptrs.size()
        );

        // Extract global_hidden from last position
        vector<float> global_hidden(HIDDEN_DIM);
        float* prefill_data = prefill_results[0].GetTensorMutableData<float>();
        memcpy(global_hidden.data(), prefill_data + (seq_len - 1) * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));

        if (args.verbose) {
            cout << "Prefill complete. global_hidden extracted (shape [" << seq_len << " x " << HIDDEN_DIM << "])." << endl;
        }

        // Extract KV cache
        vector<vector<float>> kv_cache_keys(NUM_GLOBAL_LAYERS);
        vector<vector<float>> kv_cache_values(NUM_GLOBAL_LAYERS);
        for (int layer = 0; layer < NUM_GLOBAL_LAYERS; ++layer) {
            int key_idx = layer * 2 + 1;
            int val_idx = layer * 2 + 2;
            size_t key_count = prefill_results[key_idx].GetTensorTypeAndShapeInfo().GetElementCount();
            size_t val_count = prefill_results[val_idx].GetTensorTypeAndShapeInfo().GetElementCount();
            kv_cache_keys[layer].resize(key_count);
            kv_cache_values[layer].resize(val_count);
            float* key_data = prefill_results[key_idx].GetTensorMutableData<float>();
            float* val_data = prefill_results[val_idx].GetTensorMutableData<float>();
            memcpy(kv_cache_keys[layer].data(), key_data, key_count * sizeof(float));
            memcpy(kv_cache_values[layer].data(), val_data, val_count * sizeof(float));
        }

        int64_t past_seq_len = seq_len;

        // === 4. DECODE LOOP ===
        if (args.verbose) cout << "Starting decode loop (max " << args.max_new_frames << " frames)..." << endl;

        vector<vector<int32_t>> generated_frames;

        // Track previous tokens for repetition penalty
        vector<vector<int32_t>> previous_tokens_by_channel(N_VQ);
        vector<vector<bool>> repetition_seen(N_VQ, vector<bool>(CODEBOOK_SIZE, false));

        for (int step = 0; step < args.max_new_frames; ++step) {
            // Build repetition_seen_mask [1, 16, 1024]
            int64_t rep_mask_shape[] = { 1, N_VQ, CODEBOOK_SIZE };
            vector<int32_t> rep_mask(CODEBOOK_SIZE * N_VQ, 0);
            for (int ch = 0; ch < N_VQ; ++ch) {
                for (int tok : previous_tokens_by_channel[ch]) {
                    if (tok >= 0 && tok < CODEBOOK_SIZE) {
                        rep_mask[ch * CODEBOOK_SIZE + tok] = 1;
                    }
                }
            }
            Ort::Value rep_mask_val = Ort::Value::CreateTensor<int32_t>(mem_info, rep_mask.data(), rep_mask.size(), rep_mask_shape, 3);

            // Random values for sampling
            uniform_real_distribution<float> dist(0.0f, 1.0f);
            float assistant_u = min(0.99999994f, max(0.0f, dist(rng)));

            int64_t audio_u_shape[] = { 1, N_VQ };
            vector<float> audio_u(N_VQ);
            for (int i = 0; i < N_VQ; ++i) {
                audio_u[i] = min(0.99999994f, max(0.0f, dist(rng)));
            }
            Ort::Value audio_u_val = Ort::Value::CreateTensor<float>(mem_info, audio_u.data(), audio_u.size(), audio_u_shape, 2);

            int64_t assistant_u_shape[] = { 1 };
            Ort::Value assistant_u_val = Ort::Value::CreateTensor<float>(mem_info, &assistant_u, 1, assistant_u_shape, 1);

            // Build local_fixed_sampled_frame inputs
            int64_t gh_shape[] = { 1, HIDDEN_DIM };
            Ort::Value gh_val = Ort::Value::CreateTensor<float>(mem_info, global_hidden.data(), HIDDEN_DIM, gh_shape, 2);

            const char* frame_in[] = { "global_hidden", "repetition_seen_mask", "assistant_random_u", "audio_random_u" };
            const char* frame_out[] = { "should_continue", "frame_token_ids" };
            Ort::Value frame_in_vals[] = { std::move(gh_val), std::move(rep_mask_val), std::move(assistant_u_val), std::move(audio_u_val) };

            auto frame_results = sess_local_frame.Run(
                Ort::RunOptions{nullptr},
                frame_in, frame_in_vals, 4,
                frame_out, 2
            );

            // Check if should continue
            int32_t* should_cont = frame_results[0].GetTensorMutableData<int32_t>();
            if (*should_cont == 0) {
                if (args.verbose) cout << "Got should_continue=0 at step " << step << endl;
                break;
            }

            // Extract frame tokens
            int32_t* frame_tokens = frame_results[1].GetTensorMutableData<int32_t>();
            vector<int32_t> frame;
            frame.reserve(N_VQ);
            for (int i = 0; i < N_VQ; ++i) {
                frame.push_back(frame_tokens[i]);
                previous_tokens_by_channel[i].push_back(frame_tokens[i]);
                if (frame_tokens[i] >= 0 && frame_tokens[i] < CODEBOOK_SIZE) {
                    repetition_seen[i][frame_tokens[i]] = true;
                }
            }
            generated_frames.push_back(frame);

            if (args.verbose && step < 5) {
                cout << "Step " << step << ": frame=[";
                for (int i = 0; i < 3 && i < N_VQ; ++i) cout << frame[i] << (i < 2 ? "," : "");
                cout << "...]" << endl;
            }

            // Update KV cache via decode step
            int64_t dec_in_shape[] = { 1, 1, ROW_WIDTH };
            vector<int32_t> dec_input(ROW_WIDTH, AUDIO_PAD_TOKEN);
            dec_input[0] = AUDIO_ASSISTANT_SLOT_TOKEN;
            for (int i = 0; i < N_VQ && i < (int)frame.size(); ++i) {
                dec_input[i + 1] = frame[i];
            }
            Ort::Value dec_in = Ort::Value::CreateTensor<int32_t>(mem_info, dec_input.data(), dec_input.size(), dec_in_shape, 3);

            int64_t past_len_shape[] = { 1 };
            int32_t current_past_len = (int32_t)past_seq_len;
            Ort::Value past_len = Ort::Value::CreateTensor<int32_t>(mem_info, &current_past_len, 1, past_len_shape, 1);

            vector<const char*> decode_inputs;
            vector<Ort::Value> decode_input_vals;

            decode_inputs.push_back("input_ids");
            decode_input_vals.push_back(std::move(dec_in));

            decode_inputs.push_back("past_valid_lengths");
            decode_input_vals.push_back(std::move(past_len));

            // Add KV cache - keep names in persistent vectors
            static vector<string> past_key_names;
            static vector<string> past_value_names;
            if (step == 0) {
                past_key_names.resize(NUM_GLOBAL_LAYERS);
                past_value_names.resize(NUM_GLOBAL_LAYERS);
            }
            for (int layer = 0; layer < NUM_GLOBAL_LAYERS; ++layer) {
                if (step == 0) {
                    past_key_names[layer] = "past_key_" + to_string(layer);
                    past_value_names[layer] = "past_value_" + to_string(layer);
                }
                int64_t key_shape[] = { 1, past_seq_len, NUM_GLOBAL_HEADS, HEAD_DIM };
                decode_inputs.push_back(past_key_names[layer].c_str());
                decode_input_vals.push_back(Ort::Value::CreateTensor<float>(
                    mem_info, kv_cache_keys[layer].data(), kv_cache_keys[layer].size(), key_shape, 4));

                int64_t val_shape[] = { 1, past_seq_len, NUM_GLOBAL_HEADS, HEAD_DIM };
                decode_inputs.push_back(past_value_names[layer].c_str());
                decode_input_vals.push_back(Ort::Value::CreateTensor<float>(
                    mem_info, kv_cache_values[layer].data(), kv_cache_values[layer].size(), val_shape, 4));
            }

            auto decode_results = sess_decode.Run(
                Ort::RunOptions{nullptr},
                decode_inputs.data(), decode_input_vals.data(), decode_inputs.size(),
                decode_out_ptrs.data(), decode_out_ptrs.size()
            );

            // Update global_hidden and KV cache
            float* new_hidden = decode_results[0].GetTensorMutableData<float>();
            memcpy(global_hidden.data(), new_hidden, HIDDEN_DIM * sizeof(float));

            for (int layer = 0; layer < NUM_GLOBAL_LAYERS; ++layer) {
                int idx = layer * 2 + 1;
                size_t new_count = decode_results[idx].GetTensorTypeAndShapeInfo().GetElementCount();
                kv_cache_keys[layer].resize(new_count);
                kv_cache_values[layer].resize(new_count);
                float* new_k = decode_results[idx].GetTensorMutableData<float>();
                float* new_v = decode_results[idx + 1].GetTensorMutableData<float>();
                memcpy(kv_cache_keys[layer].data(), new_k, new_count * sizeof(float));
                memcpy(kv_cache_values[layer].data(), new_v, new_count * sizeof(float));
            }
            past_seq_len += 1;
        }

        if (args.verbose) cout << "Generated " << generated_frames.size() << " frames" << endl;

        if (generated_frames.empty()) {
            cerr << "Error: No frames generated" << endl;
            return 1;
        }

        // === 5. CODEC DECODE ===
        if (args.verbose) cout << "Running codec decode..." << endl;

        int64_t num_frames = (int64_t)generated_frames.size();
        int64_t codec_shape[] = { 1, num_frames, N_VQ };
        vector<int32_t> codec_input(num_frames * N_VQ);
        for (int64_t i = 0; i < num_frames; ++i) {
            for (int64_t j = 0; j < N_VQ; ++j) {
                codec_input[i * N_VQ + j] = generated_frames[i][j];
            }
        }
        Ort::Value codec_codes = Ort::Value::CreateTensor<int32_t>(mem_info, codec_input.data(), codec_input.size(), codec_shape, 3);

        int64_t lengths_shape[] = { 1 };
        int32_t lengths_data = (int32_t)num_frames;
        Ort::Value codec_lengths = Ort::Value::CreateTensor<int32_t>(mem_info, &lengths_data, 1, lengths_shape, 1);

        const char* codec_in[] = { "audio_codes", "audio_code_lengths" };
        const char* codec_out[] = { "audio", "audio_lengths" };
        Ort::Value codec_in_vals[] = { std::move(codec_codes), std::move(codec_lengths) };

        auto codec_results = sess_codec_decode.Run(
            Ort::RunOptions{nullptr},
            codec_in, codec_in_vals, 2,
            codec_out, 2
        );

        // Get audio output
        auto audio_shape = codec_results[0].GetTensorTypeAndShapeInfo().GetShape();
        int32_t* audio_len = codec_results[1].GetTensorMutableData<int32_t>();

        float* audio_data = codec_results[0].GetTensorMutableData<float>();
        size_t total_samples = codec_results[0].GetTensorTypeAndShapeInfo().GetElementCount();

        float audio_min = *min_element(audio_data, audio_data + total_samples);
        float audio_max = *max_element(audio_data, audio_data + total_samples);

        if (args.verbose) {
            cout << "Codec output shape: [" << audio_shape[0] << ", " << audio_shape[1] << ", " << audio_shape[2] << "]" << endl;
            cout << "Audio length: " << *audio_len << " samples" << endl;
            cout << "Audio range: [" << audio_min << ", " << audio_max << "]" << endl;
        }

        // === 5. WRITE WAV ===
        auto end_total = chrono::high_resolution_clock::now();
        WavWriter::Write(args.output, audio_data, *audio_len, CODEC_SAMPLE_RATE, CODEC_CHANNELS);

        if (args.verbose) {
            auto ms = chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count();
            cout << "Generated " << args.output << " (" << ms << "ms)" << endl;
        }

    } catch (const exception& e) {
        cerr << "\n[!] Inference Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}