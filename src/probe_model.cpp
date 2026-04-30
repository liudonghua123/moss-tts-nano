#include <iostream>
#include <fstream>
#include <string>
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
std::wstring ToOrtPath(const std::string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}
#else
std::string ToOrtPath(const std::string& str) { return str; }
#endif

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
    Ort::Session local_dec(env, ToOrtPath("onnx/MOSS-TTS-Nano-100M-ONNX/moss_tts_local_decoder.onnx").c_str(), opts);
    probeModel(local_dec, "LOCAL DECODER");

    std::cout << std::endl << "Probing local_fixed_sampled_frame model..." << std::endl;
    Ort::Session local_frame(env, ToOrtPath("onnx/MOSS-TTS-Nano-100M-ONNX/moss_tts_local_fixed_sampled_frame.onnx").c_str(), opts);
    probeModel(local_frame, "LOCAL FIXED SAMPLED FRAME");

    std::cout << std::endl << "Probing codec decode model..." << std::endl;
    Ort::Session codec_dec(env, ToOrtPath("onnx/MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_decode_full.onnx").c_str(), opts);
    probeModel(codec_dec, "CODEC DECODE FULL");

    return 0;
}