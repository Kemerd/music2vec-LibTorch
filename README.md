# Music2Vec for LibTorch

A tool to convert the Music2Vec model for use with C++ LibTorch, enabling efficient music embedding generation in native code.

## Why This Tool Exists

### PyTorch vs LibTorch Limitations

PyTorch is a powerful framework for machine learning with Python, but many applications require the performance benefits of C++. LibTorch is the C++ implementation of PyTorch, but it comes with several limitations:

1. **Limited API**: LibTorch provides only a subset of PyTorch's Python API
2. **No HuggingFace Transformers support**: The extensive library of transformers models available in Python cannot be directly used in LibTorch
3. **Gradient checkpointing issues**: Features like gradient checkpointing used in model training cause issues in LibTorch inference
4. **Complex nested operations**: Many PyTorch models use dynamic Python features that don't translate well to C++

This tool addresses these limitations by converting the Music2Vec model to a format compatible with LibTorch while preserving its functionality.

### Scripting vs Tracing

This project handles model conversion using TorchScript, which can be created in two ways:

- **Tracing**: Records operations executed during a single forward pass with specific inputs
  - Pro: Simpler process
  - Con: Only captures the specific execution path for given inputs
  - Con: Doesn't capture control flow (if/else, loops)

- **Scripting**: Analyzes Python code and converts it to TorchScript
  - Pro: Captures control flow and dynamic behavior
  - Pro: More complete representation of model logic
  - Con: Not all Python features can be scripted (comprehensions with conditions, dynamic typing)

This tool uses a hybrid approach, primarily focusing on tracing but with carefully modified code to ensure compatibility.

## What This Tool Does

### Key Features

1. **Monkey Patching for LibTorch Compatibility**
   - Disables gradient checkpointing which improves execution performance
   - Removes operations that modify `requires_grad` during inference
   - Replaces problematic list comprehensions and dynamic Python operations

2. **Simplified Interface**
   - Creates a wrapper class that handles both preprocessing and inference in one step
   - Normalizes embeddings for consistent vector representation

3. **C++ Integration Ready**
   - Exports a model that can be loaded directly in C++ code
   - Includes configuration files for reference

## Usage Instructions

### Converting the Model

1. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Run the Conversion Script**
   ```
   # On Windows
   convert.bat
   
   # On Linux/Mac
   python convert.py
   ```

3. **Verify Output**
   The script will create a directory with the exported model and configuration files.

### Using the Model in C++

```cpp
#include <torch/script.h>
#include <vector>
#include <iostream>
#include <cmath>

// Generate a sine wave audio signal
torch::Tensor generate_sine_wave(float frequency, float duration, int sample_rate = 16000) {
    int num_samples = static_cast<int>(duration * sample_rate);
    torch::Tensor time = torch::arange(0, num_samples) / sample_rate;
    
    // Generate sine wave: amplitude * sin(2π * frequency * time)
    float amplitude = 0.5;  // Avoid clipping
    torch::Tensor audio = amplitude * torch::sin(2 * M_PI * frequency * time);
    
    // Add batch dimension [1, audio_length]
    return audio.unsqueeze(0);
}

// Calculate cosine similarity between two tensors
float cosine_similarity(const torch::Tensor& a, const torch::Tensor& b) {
    // Compute dot product
    torch::Tensor dot_product = torch::sum(a * b);
    
    // Compute norms
    torch::Tensor norm_a = torch::sqrt(torch::sum(a * a));
    torch::Tensor norm_b = torch::sqrt(torch::sum(b * b));
    
    // Return cosine similarity
    return dot_product.item<float>() / (norm_a.item<float>() * norm_b.item<float>());
}

int main() {
    try {
        // Load the model
        torch::jit::script::Module model = torch::jit::load("path/to/model.pt");
        model.eval();
        
        // Generate two different sine waves (3 seconds each)
        // 440 Hz = A4 note
        torch::Tensor audio1 = generate_sine_wave(440.0, 3.0);  
        // 880 Hz = A5 note (one octave higher)
        torch::Tensor audio2 = generate_sine_wave(880.0, 3.0);
        
        // Perform inference on first sine wave
        std::vector<torch::jit::IValue> inputs1;
        inputs1.push_back(audio1);
        torch::Tensor embedding1 = model.forward(inputs1).toTensor();
        
        // Perform inference on second sine wave
        std::vector<torch::jit::IValue> inputs2;
        inputs2.push_back(audio2);
        torch::Tensor embedding2 = model.forward(inputs2).toTensor();
        
        // Print the embedding shapes
        std::cout << "Embedding shape: " 
                  << embedding1.sizes()[0] << " x " 
                  << embedding1.sizes()[1] << std::endl;
        
        // Compare the embeddings
        float similarity = cosine_similarity(embedding1.squeeze(), embedding2.squeeze());
        std::cout << "Similarity between A4 and A5 notes: " << similarity << std::endl;
        
        // Values closer to 1 indicate higher similarity
        // This is useful for music similarity search, genre classification, etc.
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
```

### Performance Considerations

- The converted model is optimized for inference performance
- Disabling gradient checkpointing significantly improves execution speed
- The model produces normalized embeddings of fixed size regardless of input audio length

## Model Details

The Music2Vec model creates 768-dimensional embeddings for music audio. It's based on the Data2VecAudio architecture and has been fine-tuned specifically for music understanding.

- **Input**: Raw audio waveform at 16kHz sampling rate
- **Output**: 768-dimensional normalized embedding vector
- **Model Architecture**: Data2Vec Audio (Transformer-based)

## License

MIT License

Copyright (c) 2023

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 