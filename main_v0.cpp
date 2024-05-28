#include <iostream>
#include<memory>
#include <torch/torch.h>
#include "onnxruntime_cxx_api.h"
#include <sndfile.h>

constexpr const int frame_size = 480;
//constexpr const int frame_size = 191488;
constexpr const int fft_size = 960;

class TorchDFPipeline {
public:
    TorchDFPipeline() {
    }

    std::vector<float> Infer(const std::string& model_path,
                             const std::vector<float>& input_audio, int sample_rate) {
        // Assuming that only one ONNX model in the pipeline for simplicity
        // Prepare input tensor

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Initialize session
        Ort::Session session = Ort::Session(env, model_path.c_str(), session_options);
        size_t input_tensor_size = input_audio.size();

        std::cout << typeof(input_audio) << std::endl;

    //    std::vector<int64_t> input_tensor_shape = {1, 1, static_cast<int64_t>(input_tensor_size)};
        std::vector<int64_t> input_tensor_shape = {static_cast<int64_t>(input_tensor_size)};
        std::vector<float> input_tensor_values(input_audio.begin(), input_audio.end());

        // Create input tensor object from data values
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_size, input_tensor_shape.data(),
                input_tensor_shape.size());

        // Prepare input names and output names
        const char* input_names[] = {"input_frame", "states", "atten_lim_db"};
        const char* output_names[] = {"enhanced_audio_frame", "out_states", "lsnr"};

        // Run inference
        
        std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                          &input_tensor, 1, output_names, 3);


          return std::vector<float>{4.3};
    }

};


// Function to read WAV file and return audio data as vector of floats
std::vector<float> load_and_preprocess_audio(const std::string& filename) {

    int sampleRate, numSamples;
    SF_INFO sfInfo;
    SNDFILE* sndFile = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    if (!sndFile) {
        std::cerr << "Error opening input file: " << sf_strerror(sndFile) << std::endl;
        exit(EXIT_FAILURE);
    }
    sampleRate = sfInfo.samplerate;
    numSamples = sfInfo.frames * sfInfo.channels;
    std::vector<float> audioData(numSamples);
    sf_read_float(sndFile, audioData.data(), numSamples);
    sf_close(sndFile);
    return audioData;
}

int main() {
    const std::string model_path = "/Users/sohamdatta/CLionProjects/audioCPPNoiseRemoval/denoiser_model.onnx";
    TorchDFPipeline pipeline{};

    // Load audio using your preferred audio library and convert to a single mono channel
    std::vector<float> audio_data = load_and_preprocess_audio("/Users/sohamdatta/CLionProjects/audioCPPNoiseRemoval/noiserRemovalSample1.wav");

    // Hardcoded sample_rate for demonstration purposes
    int sample_rate = 48000;

    // Run inference
    std::vector<float> enhanced_audio = pipeline.Infer(model_path, audio_data, sample_rate);

    // Save the enhanced audio using your preferred audio library
    // save_audio("/path/to/your/enhanced_audio.wav", enhanced_audio, sample_rate);
    return 0;
}
