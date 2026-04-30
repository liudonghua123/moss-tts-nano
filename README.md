# MOSS-TTS-Nano C++

A high-performance C++ implementation of MOSS-TTS-Nano voice cloning text-to-speech system using ONNX Runtime.

## Project Overview

MOSS-TTS-Nano is a lightweight voice cloning TTS model that can synthesize speech in any voice from just a short audio prompt. This project provides a native C++ implementation optimized for:

- **CPU inference** with ONNX Runtime
- **Voice cloning** from reference audio
- **Built-in voices** with pre-extracted prompt codes
- **Cross-platform support** (Windows, Linux, macOS)

### Key Features

- Voice cloning with reference audio
- Multiple built-in voices (currently: Junhao - Chinese Male)
- WAV output at 48kHz stereo
- Fast CPU inference without GPU dependency
- Batch processing support

## Quick Start

### Prerequisites

- CMake 3.10+
- C++ compiler with C++17 support (MSVC, GCC, Clang)
- ONNX Runtime 1.16+ (automatically downloaded)
- SentencePiece (automatically downloaded)

### Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

The executable will be at `build/Release/Release/moss-tts-nano.exe` (Windows) or `build/Release/moss-tts-nano` (Linux/macOS).

### Download Models (if not using release package)

Download the ONNX models from HuggingFace using `huggingface-cli` or `hf`:

```bash
# Install huggingface-cli
pip install huggingface-hub

# Create directories
mkdir -p onnx

# Download TTS model (MOSS-TTS-Nano-100M)
hf download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX --local-dir onnx/MOSS-TTS-Nano-100M-ONNX

# Download Codec model (MOSS-Audio-Tokenizer-Nano)
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX --local-dir onnx/MOSS-Audio-Tokenizer-Nano-ONNX

# Create voices directory for custom voices
mkdir -p voices
```

### Usage

**Basic usage (using built-in Junhao voice):**
```bash
./build/Release/Release/moss-tts-nano --text "你好，这是一个测试。" --out output.wav
```

**Voice cloning with reference audio (direct path):**
```bash
./build/Release/Release/moss-tts-nano --text "你好，这是一个测试。" --out output.wav --voice /path/to/reference.wav
```

**Voice cloning with built-in voice name (from voices/ directory):**
```bash
./build/Release/Release/moss-tts-nano --text "你好，这是一个测试。" --out output.wav --voice junhao
```

**Create voices directory and add custom voices:**
```bash
mkdir voices
# Place your reference audio as voices/your_voice.wav
./build/Release/Release/moss-tts-nano --text "你好" --voice your_voice --out output.wav
```

**Custom model directories:**
```bash
./build/Release/Release/moss-tts-nano \
    --text "你好，这是一个测试。" \
    --out output.wav \
    --model-dir onnx/custom-tts-dir \
    --codec-dir onnx/custom-codec-dir
```

**Adjust generation length:**
```bash
./build/Release/Release/moss-tts-nano --text "这是一个很长的句子。" --out output.wav --max-frames 200
```

## Technical Implementation

### Model Architecture

The MOSS-TTS-Nano system consists of several ONNX models:

1. **Prefill Model** (`moss_tts_prefill.onnx`)
   - Processes the input text + prompt tokens
   - Outputs global hidden states and initial KV cache

2. **Decode Step Model** (`moss_tts_decode_step.onnx`)
   - Autoregressive generation step
   - Updates KV cache with new tokens
   - Produces global hidden states for local decoder

3. **Local Decoder Model** (`moss_tts_local_decoder.onnx`)
   - WaveNet-style architecture for audio token prediction
   - Predicts text and audio logits from global hidden states

4. **Local Fixed Sampled Frame Model** (`moss_tts_local_fixed_sampled_frame.onnx`)
   - Sampling with repetition penalty
   - Decides whether to continue generation
   - Outputs frame token IDs

5. **Codec Encode Model** (`moss_audio_tokenizer_encode.onnx`)
   - Encodes audio to discrete tokens
   - Used for voice cloning from reference audio

6. **Codec Decode Model** (`moss_audio_tokenizer_decode_full.onnx`)
   - Converts discrete audio tokens back to waveform
   - Outputs 48kHz stereo audio

### Pipeline

```
Input Text
    │
    ▼
┌─────────────────┐
│ Tokenize (BPE)  │
└─────────────────┘
    │
    ▼
┌─────────────────┐     ┌──────────────────────────┐
│ Build Prompt    │────►│ USER_PROMPT + REF_AUDIO + │
│ Template        │     │ TEXT + ASSISTANT_PROMPT   │
└─────────────────┘     └──────────────────────────┘
    │                            │
    ▼                            ▼
┌─────────────────┐     ┌──────────────────────────┐
│ Prefill Model   │────►│ Global Hidden States +    │
│                 │     │ Initial KV Cache          │
└─────────────────┘     └──────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ Local     │   │ Fixed     │   │ Decode    │
            │ Decoder   │──►│ Sampled   │──►│ Step      │
            │           │   │ Frame     │   │ (KV Cache)│
            └───────────┘   └───────────┘   └───────────┘
                    │               │
                    │   ┌───────────┴───┐
                    │   │ Audio Tokens  │
                    │   │ (16 channels) │
                    │   └───────────────┘
                    ▼                   ▼
            ┌───────────────────────────────────┐
            │ Codec Decode → 48kHz Stereo WAV  │
            └───────────────────────────────────┘
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_VQ` | 16 | Number of VQ codebooks |
| `ROW_WIDTH` | 17 | Input width (1 text token + 16 audio tokens) |
| `HIDDEN_DIM` | 768 | Model hidden dimension |
| `VOCAB_SIZE` | 16384 | Text vocabulary size |
| `CODEBOOK_SIZE` | 1024 | Audio codebook size (10-bit) |
| `CODEC_SAMPLE_RATE` | 48000 | Output audio sample rate |
| `CODEC_CHANNELS` | 2 | Output channels (stereo) |
| `NUM_GLOBAL_LAYERS` | 12 | Transformer layers |
| `NUM_GLOBAL_HEADS` | 12 | Attention heads |
| `HEAD_DIM` | 64 | Attention head dimension |

### Special Tokens

| Token | ID | Description |
|-------|-----|-------------|
| `AUDIO_PAD_TOKEN` | 1024 | Padding for audio tokens |
| `AUDIO_START_TOKEN` | 6 | Marks start of audio generation |
| `AUDIO_END_TOKEN` | 7 | Marks end of audio generation |
| `AUDIO_USER_SLOT_TOKEN` | 8 | Marks user-provided audio in prompt |
| `AUDIO_ASSISTANT_SLOT_TOKEN` | 9 | Marks assistant audio to generate |

## Code Structure

```
src/
├── main.cpp           # Main implementation
├── wav_writer.h       # WAV file writing utility
└── probe_model.cpp    # ONNX model inspection tool
```

### main.cpp

| Section | Lines | Description |
|---------|-------|-------------|
| Includes & Utils | 1-28 | Headers and path conversion |
| Constants | 30-70 | Model parameters and special tokens |
| Prompt Templates | 59-84 | Built-in voice prompt codes |
| Helper Functions | 86-136 | Row building utilities |
| Main Function | 138-620 | Full TTS pipeline |

### Key Functions

- `buildTextRows()` - Converts token IDs to model input rows
- `buildAudioPrefixRows()` - Builds audio prefix from prompt codes
- `buildVoiceCloneRequestRows()` - Combines all components into complete request
- `ToOrtPath()` - Platform-specific path conversion for ONNX Runtime

### wav_writer.h

Writes 16-bit PCM WAV files with proper headers:
- Supports mono and stereo output
- Converts float32 [-1, 1] to int16 range
- Proper RIFF/WAVE chunk formatting

## Comparison with Python Reference

This C++ implementation is a native port of the Python reference implementation (`python-references/ort_cpu_runtime.py`). Here are the key differences:

| Feature | Python Reference | C++ Implementation |
|---------|-----------------|-------------------|
| **Runtime** | ONNX Runtime Python API | ONNX Runtime C++ API |
| **Dependencies** | Python 3.8+, onnxruntime | C++17, CMake |
| **Binary Size** | N/A (interpreter required) | ~1MB executable |
| **Memory Usage** | Python overhead | Minimal overhead |
| **Speed** | Similar | Similar (CPU-bound) |
| **Platform** | Cross-platform (Python) | Cross-platform (native) |
| **Deployment** | Requires Python runtime | Self-contained binary |
| **Voice Cloning** | Via prompt audio encoding | Via prompt audio encoding |
| **Batch Processing** | Supported | Single sequence (v1) |

### Design Decisions

1. **Single Binary Distribution**: The C++ version bundles all required ONNX models into a single release package, simplifying deployment.

2. **Built-in Voice Support**: Unlike the Python version which requires encoding prompt audio every time, the C++ version includes pre-computed prompt codes for the default "junhao" voice, reducing initialization time.

3. **CPU-Optimized**: Both implementations target CPU inference. GPU acceleration can be added via ONNX Runtime's CUDA provider.

### API Differences

**Python (reference):**
```python
# Requires manual prompt encoding
prompt_codes = encode_audio(prompt_waveform)  # Must call encoder first
request = build_voice_clone_request(prompt_codes, text_tokens)
result = run_tts(request)
```

**C++ (this implementation):**
```bash
# Simple command-line interface
moss-tts-nano --text "你好" --voice junhao --out output.wav
# Or with custom voice
moss-tts-nano --text "你好" --voice my_voice.wav --out output.wav
```

## Troubleshooting

**"No input" error:**
```bash
# Must provide --text argument
./moss-tts-nano --text "你好"
```

**Model files not found:**
- Verify ONNX models are in correct directories
- Check model paths match `--model-dir` and `--codec-dir` arguments

**Low audio volume:**
- Output is normalized to [-1, 1] range
- Use an audio editor to boost volume if needed

**Slow generation:**
- CPU inference is expected; generation takes 30-60 seconds for 100 frames
- Reduce `--max-frames` for faster but shorter output

## Performance

- **CPU**: Intel/AMD x86-64, ARM64
- **Memory**: ~2GB RAM for models
- **Generation speed**: ~1-2 frames/second on modern CPU
- **Output**: 48kHz stereo WAV

## License

This project follows the same license as the original MOSS-TTS-Nano project.

## Acknowledgments

- [FengJunhao/MOSS-TTS-Nano](https://huggingface.co/fengjunhao/MOSS-TTS-Nano-100M-ONNX) - Original model weights
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform ML inference
- [SentencePiece](https://github.com/google/sentencepiece) - Tokenization