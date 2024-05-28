
# DeepfilterNet Onnx model inference in C++

This repo contains the C++ inference code of DeepFilteNet onnx model. 

**libsndfile** is required to be installed to read and write "wav" file

**Install**:
```bash
brew install libsndfile
```

**Build:**

```bash
cd build 
cmake --build . --config Release
```

If CMakeLists.txt is modified, the build process will be as follows
```bash
cd build 
cmake ..
cmake -build . --config Release
```

**Run** 

Inside build folder 
```bash
./audioCPPNoiseRemoval <onnx_model_path> <input_audio_path>
<enhanced_audio_path>
```

*Use absolute paths

For example, 
```bash

./audioCPPNoiseRemoval /Users/folder/audioCPPNoiseRemoval/denoiser_model.onnx /Users/folder/audioCPPNoiseRemoval/Kal_Ho_Na_ho_Deb.wav ../enhanced_audio.wav
```

Note: The input audio files should be in "wav" format and sample rate should be 48000 Hz. If the sampling rate is not 48000 Hz, the resample code in resample.py in the repo can be used to upsample the audio file. 


**Future Work**

1. Curerntly audio files are shortened to 30 secs to facilitate low latency. Online streaming can be used to improve the latency.
2. Need to debug segmentation fault for longer files appearing in some cases. 