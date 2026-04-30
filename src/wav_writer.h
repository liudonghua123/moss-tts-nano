#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

class WavWriter {
public:
    static void Write(const std::string& filename, const float* data, size_t num_samples, int sample_rate, int num_channels = 1) {
        std::ofstream file(filename, std::ios::binary);

        // Header
        file.write("RIFF", 4);
        uint32_t file_size = 36 + num_samples * 2 * num_channels;
        file.write(reinterpret_cast<char*>(&file_size), 4);
        file.write("WAVE", 4);

        // Format chunk
        file.write("fmt ", 4);
        uint32_t fmt_size = 16;
        uint16_t audio_format = 1; // PCM
        uint16_t n_channels = (uint16_t)num_channels;
        uint32_t s_rate = sample_rate;
        uint32_t byte_rate = sample_rate * 2 * num_channels;
        uint16_t block_align = 2 * num_channels;
        uint16_t bits_per_sample = 16;

        file.write(reinterpret_cast<char*>(&fmt_size), 4);
        file.write(reinterpret_cast<char*>(&audio_format), 2);
        file.write(reinterpret_cast<char*>(&n_channels), 2);
        file.write(reinterpret_cast<char*>(&s_rate), 4);
        file.write(reinterpret_cast<char*>(&byte_rate), 4);
        file.write(reinterpret_cast<char*>(&block_align), 2);
        file.write(reinterpret_cast<char*>(&bits_per_sample), 2);

        // Data chunk
        file.write("data", 4);
        uint32_t data_size = num_samples * 2 * num_channels;
        file.write(reinterpret_cast<char*>(&data_size), 4);

        // Write samples (convert float [-1, 1] to int16)
        // Audio data is in format [channels, samples] or [1, samples] for mono
        if (num_channels == 1) {
            for (size_t i = 0; i < num_samples; ++i) {
                float s = data[i];
                if (s > 1.0f) s = 1.0f;
                if (s < -1.0f) s = -1.0f;
                int16_t sample = static_cast<int16_t>(s * 32767.0f);
                file.write(reinterpret_cast<char*>(&sample), 2);
            }
        } else {
            // Interleaved stereo: channel-major input [channels, samples] -> interleaved output
            for (size_t i = 0; i < num_samples; ++i) {
                for (int ch = 0; ch < num_channels; ++ch) {
                    float s = data[ch * num_samples + i];
                    if (s > 1.0f) s = 1.0f;
                    if (s < -1.0f) s = -1.0f;
                    int16_t sample = static_cast<int16_t>(s * 32767.0f);
                    file.write(reinterpret_cast<char*>(&sample), 2);
                }
            }
        }
    }
};