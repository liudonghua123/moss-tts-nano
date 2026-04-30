#include <iostream>
#include <fstream>
#include <onnxruntime_cxx_api.h>

void probeModel(Ort::Session& sess, const char* name) {
    std::cout << std::endl << "=== " << name << " ===" << std::endl;
    std::cout << "Inputs:" << std::endl;
    size_t in_count = sess.GetInputCount();
    for (size_t i = 0; i < in_count; ++i) {
        auto n = sess.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        auto type_info = sess.GetInputTypeInfo(i);
        auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        auto type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
        std::cout << "  [" << i << "] " << n.get() << " shape: ";
        for (size_t s = 0; s < shape.size(); ++s) {
            std::cout << shape[s] << (s < shape.size()-1 ? " x " : "");
        }
        std::cout << " type:" << type << std::endl;
    }
    std::cout << "Outputs:" << std::endl;
    size_t out_count = sess.GetOutputCount();
    for (size_t i = 0; i < out_count; ++i) {
        auto n = sess.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        auto type_info = sess.GetOutputTypeInfo(i);
        auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        auto type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
        std::cout << "  [" << i << "] " << n.get() << " shape: ";
        for (size_t s = 0; s < shape.size(); ++s) {
            std::cout << shape[s] << (s < shape.size()-1 ? " x " : "");
        }
        std::cout << " type:" << type << std::endl;
    }
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelProbe");
    Ort::SessionOptions opts;

    std::cout << "Probing local_decoder model..." << std::endl;
    Ort::Session local_dec(env, L"onnx/MOSS-TTS-Nano-100M-ONNX/moss_tts_local_decoder.onnx", opts);
    probeModel(local_dec, "LOCAL DECODER");

    std::cout << std::endl << "Probing local_fixed_sampled_frame model..." << std::endl;
    Ort::Session local_frame(env, L"onnx/MOSS-TTS-Nano-100M-ONNX/moss_tts_local_fixed_sampled_frame.onnx", opts);
    probeModel(local_frame, "LOCAL FIXED SAMPLED FRAME");

    std::cout << std::endl << "Probing codec decode model..." << std::endl;
    Ort::Session codec_dec(env, L"onnx/MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_decode_full.onnx", opts);
    probeModel(codec_dec, "CODEC DECODE FULL");

    return 0;
}