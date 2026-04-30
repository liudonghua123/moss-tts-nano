#!/bin/bash
mkdir -p models
BASE_TTS="https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX/resolve/main"
BASE_VOC="https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX/resolve/main"

echo "Downloading MOSS-TTS-Nano Models..."
# TTS Generator Files
curl -L "$BASE_TTS/moss_tts_prefill.onnx" -o models/moss_tts_prefill.onnx
curl -L "$BASE_TTS/moss_tts_decode_step.onnx" -o models/moss_tts_decode_step.onnx
curl -L "$BASE_TTS/moss_tts_global_shared.data" -o models/moss_tts_global_shared.data
curl -L "$BASE_TTS/moss_tts_local_shared.data" -o models/moss_tts_local_shared.data
curl -L "$BASE_TTS/tokenizer.model" -o models/tokenizer.model

# Audio Tokenizer (Vocoder) Files
curl -L "$BASE_VOC/moss_audio_tokenizer_decode_full.onnx" -o models/vocoder.onnx
curl -L "$BASE_VOC/moss_audio_tokenizer_decode_shared.data" -o models/moss_audio_tokenizer_decode_shared.data

echo "Done. Ensure all .data files remain next to their .onnx counterparts."