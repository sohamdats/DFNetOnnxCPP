#include <iostream>
#include<memory>
// #include <torch/torch.h>
#include "onnxruntime_cxx_api.h"
#include <sndfile.h>

constexpr const int frame_size = 480;
constexpr const int fft_size = 960;

class TorchDFPipeline {
public:
    TorchDFPipeline() {
    }
    Ort::Session getSession(const std::string& model_path){

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Initialize session
        Ort::Session session = Ort::Session(env, model_path.c_str(), session_options);
        return session;

    }

    void pad(std::vector<float>& input_audio, int pad_left, int pad_right) {
            input_audio.insert(input_audio.begin(), pad_left, 0.0f); // Pad left
            input_audio.insert(input_audio.end(), pad_right, 0.0f); // Pad right
        }

    std::vector<float> tensor_to_vector(const Ort::Value& value) {
            
             if (!value.IsTensor()) {
        throw std::invalid_argument("The provided Ort::Value is not a tensor.");
    }

    // Get the tensor's data type
            auto type_info = value.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType data_type = type_info.GetElementType();
            if (data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                throw std::invalid_argument("The tensor does not contain float data.");
            }

            // Get the total number of elements in the tensor
            size_t total_elements = type_info.GetElementCount();

            // Get a pointer to the tensor's data
            const float* tensor_data = value.GetTensorData<float>();

            // Create a std::vector and copy the data
            std::vector<float> result(tensor_data, tensor_data + total_elements);

            return result;
        }

    std::vector<float> Infer(const std::string& model_path,
                            std::vector<float>& input_audio) {
        // Assuming that only one ONNX model in the pipeline for simplicity
        // Prepare input tensor

        Ort::Session session = getSession(model_path);
        
        std::vector<float> states(45304, 0.0);
        std::vector<float> atten_lim_db = {0.0};
        
        // Create input tensor object from data values
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        

        size_t states_tensor_size = states.size();
        std::vector<int64_t> states_tensor_shape = {static_cast<int64_t>(states_tensor_size)};
        std::vector<float> states_tensor_values(states.begin(), states.end());

        Ort::Value states_tensor = Ort::Value::CreateTensor<float>(
                memory_info, states_tensor_values.data(), states_tensor_size, states_tensor_shape.data(),
                states_tensor_shape.size());

        
        size_t atten_lim_db_tensor_size = atten_lim_db.size();
        std::vector<int64_t> atten_lim_db_tensor_shape = {static_cast<int64_t>(atten_lim_db_tensor_size)};
        std::vector<float> atten_lim_db_tensor_values(atten_lim_db.begin(), atten_lim_db.end());

        Ort::Value atten_lim_db_tensor = Ort::Value::CreateTensor<float>(
                memory_info, atten_lim_db_tensor_values.data(), atten_lim_db_tensor_size, atten_lim_db_tensor_shape.data(),
                atten_lim_db_tensor_shape.size());

        // chunking the input
        
        int hop_size = 480;
        int orig_len = input_audio.size();
        int fft_size = 960;

        // Calculate hop_size_divisible_padding_size
        int hop_size_divisible_padding_size = (hop_size - orig_len % hop_size) % hop_size;
        orig_len += hop_size_divisible_padding_size;


        // Padding using F.pad
        pad(input_audio, 0, fft_size + hop_size_divisible_padding_size);
    
        // Splitting into chunks

        const char* input_names[] = {"input_frame", "states", "atten_lim_db"};
        const char* output_names[] = {"enhanced_audio_frame", "out_states", "lsnr"};

        int num_chunks = input_audio.size() / hop_size;

        std::cout<< "No of chunks: " << num_chunks << std::endl;

        // std::vector<float> chunk_audio

        std::vector<Ort::Value> output_frames;

        int allowed_nchunks = 3270;
        
        if (num_chunks > allowed_nchunks){

            std::cout << "\nAudio file is shortened to 30 seconds to avoid high latency" << std::endl;
            num_chunks = allowed_nchunks;
            std::cout<< "New count of chunks: " << num_chunks << std::endl;
       
        }
            
        for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {

            auto start = input_audio.begin() + chunk_id* hop_size;
            auto end = start + hop_size;

            std::vector<float> input_frame(start, end);

            size_t input_tensor_size = input_frame.size();

            std::vector<int64_t> input_tensor_shape = {static_cast<int64_t>(input_tensor_size)};
            std::vector<float> input_tensor_values(input_frame.begin(), input_frame.end());

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_tensor_values.data(), input_tensor_size, input_tensor_shape.data(),
                    input_tensor_shape.size());


            Ort::Value atten_lim_db_tensor = Ort::Value::CreateTensor<float>(
                memory_info, atten_lim_db_tensor_values.data(), atten_lim_db_tensor_size, atten_lim_db_tensor_shape.data(),
                atten_lim_db_tensor_shape.size());

            Ort::Value input_tensors[3] = {std::move(input_tensor), std::move(states_tensor), std::move(atten_lim_db_tensor)};

            // Run inference            
            std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                            input_tensors , 3, output_names, 3);

            output_frames.push_back(std::move(output_tensors[0]));
            states_tensor = std::move(output_tensors[1]);

        }

        std::vector<float> output_frames_vector;
        for(Ort::Value& tensor: output_frames){

            std::vector<float> enhanced_chunk = tensor_to_vector(tensor);
            output_frames_vector.insert(output_frames_vector.end(), enhanced_chunk.begin(), enhanced_chunk.end());
        }
  
        int d = fft_size - hop_size;
        std::vector<float> final_audio(output_frames_vector.begin()+d, output_frames_vector.begin()+orig_len+d);

        return final_audio;
    }

};

// Function to read WAV file and return audio data as vector of floats
std::vector<float> load_and_preprocess_audio(const std::string& filename) {

    int sampleRate, numSamples, channels;
    SF_INFO sfInfo;
    SNDFILE* sndFile = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    if (!sndFile) {
        std::cerr << "Error opening input file: " << sf_strerror(sndFile) << std::endl;
        exit(EXIT_FAILURE);
    }
    sampleRate = sfInfo.samplerate;
    channels = sfInfo.channels;
    std::cout << "While reading sr: " << sampleRate << std::endl;
    std::cout << "while reading channels: " << sfInfo.channels << std::endl;
    
    numSamples = sfInfo.frames * sfInfo.channels;
    std::vector<float> audioData(numSamples);


    sf_read_float(sndFile, audioData.data(), numSamples);
    sf_close(sndFile);
    return audioData;
}

void write_audio_to_wav(const std::string& filename, const std::vector<float>& audioData, int sampleRate, int channels) {
    // Define the format of the WAV file
    SF_INFO sfInfo;
    sfInfo.samplerate = sampleRate;
    sfInfo.channels = channels;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    // Open the output file
    SNDFILE* sndFile = sf_open(filename.c_str(), SFM_WRITE, &sfInfo);
    if (!sndFile) {
        std::cerr << "Error opening output file: " << sf_strerror(sndFile) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Write the audio data to the file
    sf_count_t numFrames = audioData.size() / channels;
    sf_write_float(sndFile, audioData.data(), audioData.size());

    // Close the file
    sf_close(sndFile);
}

int main(int argc, char* argv[]) {
    
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << "<audio_path> <output_filename>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string audio_path = argv[2];
    std::string output_filename = argv[3];
    
    TorchDFPipeline pipeline{};

    // Load audio using your preferred audio library and convert to a single mono channel
    std::vector<float> audio_data = load_and_preprocess_audio(audio_path);

    // Hardcoded sample_rate for demonstration purpose
    int sample_rate = 48000;

    // Run inference
    std::vector<float> enhanced_audio_vector = pipeline.Infer(model_path, audio_data);

    write_audio_to_wav(output_filename, enhanced_audio_vector, sample_rate, 1);
    return 0;
}
