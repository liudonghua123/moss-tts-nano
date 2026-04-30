@echo off
SETLOCAL EnableDelayedExpansion

:: Configuration
set MODEL_DIR=models
set BASE_TTS=https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX/resolve/main
set BASE_VOC=https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX/resolve/main

echo ====================================================
echo MOSS-TTS-Nano Asset Downloader
echo ====================================================

if not exist %MODEL_DIR% (
    mkdir %MODEL_DIR%
    echo Created directory: %MODEL_DIR%
)

echo.
echo [1/2] Downloading TTS Generator Models...
:: Prefill Model
curl -L "%BASE_TTS%/moss_tts_prefill.onnx" -o "%MODEL_DIR%/moss_tts_prefill.onnx"
curl -L "%BASE_TTS%/moss_tts_global_shared.data" -o "%MODEL_DIR%/moss_tts_global_shared.data"
:: Decode Step Model (Optional for basic CLI, but good to have)
curl -L "%BASE_TTS%/moss_tts_decode_step.onnx" -o "%MODEL_DIR%/moss_tts_decode_step.onnx"
curl -L "%BASE_TTS%/moss_tts_local_shared.data" -o "%MODEL_DIR%/moss_tts_local_shared.data"
curl -L "%BASE_TTS%/tokenizer.model" -o "%MODEL_DIR%/tokenizer.model"

echo.
echo [2/2] Downloading Audio Tokenizer (Vocoder)...
curl -L "%BASE_VOC%/moss_audio_tokenizer_decode_full.onnx" -o "%MODEL_DIR%/vocoder.onnx"
curl -L "%BASE_VOC%/moss_audio_tokenizer_decode_shared.data" -o "%MODEL_DIR%/moss_audio_tokenizer_decode_shared.data"

echo.
echo ====================================================
echo Model setup complete.
echo Ensure the .data files stay in the same folder as the .onnx files.
echo ====================================================
pause