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

int main() {
    try {
        // Load the model
        torch::jit::script::Module model = torch::jit::load("path/to/model.pt");
        model.eval();
        
        // Create input tensor (audio waveform at 16kHz)
        // This example uses 1 second of silence
        torch::Tensor audio = torch::zeros({1, 16000});
        
        // Perform inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(audio);
        at::Tensor embedding = model.forward(inputs).toTensor();
        
        // Print the shape of the embedding
        std::cout << "Embedding shape: " 
                  << embedding.sizes()[0] << " x " 
                  << embedding.sizes()[1] << std::endl;
                  
        // Embedding can now be used for music similarity, search, etc.
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