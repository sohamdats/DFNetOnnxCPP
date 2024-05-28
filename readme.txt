cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release

model_path = "/Users/sohamdatta/CLionProjects/audioCPPNoiseRemoval/denoiser_model.onnx"
input_audio = "/Users/sohamdatta/CLionProjects/audioCPPNoiseRemoval/noiserRemocalSample1resampled.wav"
output_audio = "/Users/sohamdatta/CLionProjects/audioCPPNoiseRemoval/enhanced_audio.wav"


./audioCPPNoiseRemoval /Users/sohamdatta/CLionProjects/audioCPPNoiseRemoval/denoiser_model.onnx /Users/sohamdatta/CLionProjects/audioCPPNoiseRemoval/Kal_Ho_Na_ho_Deb.wav ../enhanced_audio.wav

./audioCPPNoiseRemoval /Users/sohamdatta/CLionProjects/audioCPPNoiseRemoval/denoiser_model.onnx /Users/sohamdatta/CLionProjects/audioCPPNoiseRemoval/sample1.wav ../enhanced_audio.wav

