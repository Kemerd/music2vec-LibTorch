"""
Convert Music2Vec Hugging Face model to TorchScript format for C++ LibTorch.

This script:
1. Downloads and loads the music2vec-v1 model from Hugging Face
2. Creates a wrapper class that handles preprocessing and inference
3. Exports the model to TorchScript format via tracing
4. Saves the traced model to the specified output directory

The resulting model can be loaded directly in C++ with LibTorch.
"""

import os
import torch
from torch import nn
from transformers import Wav2Vec2Processor, Data2VecAudioModel
import shutil
import types

# Output directory - specifically in third_party folder
OUTPUT_DIR = "third_party/music2vec-v1_c"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Will save model to: {os.path.abspath(OUTPUT_DIR)}")

print("Loading processor and model from Hugging Face...")
# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
model = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1")

# Set model to evaluation mode
model.eval()
# Disable gradient checkpointing to avoid requires_grad issues during scripting
if hasattr(model, "gradient_checkpointing"):
    model.gradient_checkpointing = False
# Disable requires_grad for all parameters
for param in model.parameters():
    param.requires_grad = False

# Fix the problematic feature encoder - the key issue is line 294 where it sets requires_grad on hidden_states
print("Patching feature encoder forward method to remove requires_grad setting...")
# Original forward method from Data2VecAudioFeatureEncoder
original_forward = model.feature_extractor.forward

# Define a new forward method that skips setting requires_grad
def patched_forward(self, input_values):
    hidden_states = input_values[:, None]
    
    # The problematic code was here:
    # if self._requires_grad and self.training:
    #    hidden_states.requires_grad = True
    
    # Continue with the rest of the original implementation
    for conv_layer in self.conv_layers:
        hidden_states = conv_layer(hidden_states)

    return hidden_states

# Apply the monkey patch
model.feature_extractor.forward = types.MethodType(patched_forward, model.feature_extractor)

# Additionally, for safety, set _requires_grad to False for feature extractor
if hasattr(model.feature_extractor, "_requires_grad"):
    print("Setting _requires_grad to False on feature extractor")
    model.feature_extractor._requires_grad = False

# Check all modules for _requires_grad and set to False
for name, module in model.named_modules():
    if hasattr(module, "_requires_grad"):
        print(f"Setting _requires_grad to False on {name}")
        module._requires_grad = False

# Fix the comprehension in the encoder's forward method
print("Patching encoder forward method to avoid comprehension with conditions...")

# Store original methods for reference
if hasattr(model, "encoder") and hasattr(model.encoder, "forward"):
    original_encoder_forward = model.encoder.forward
    
    # Define patched encoder forward method to avoid list comprehension with conditions
    def patched_encoder_forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Patched version of the encoder's forward method that avoids using list comprehensions with conditions.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states[~attention_mask] = 0
            
            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = 0
            skip_the_layer = False  # in inference mode we don't skip layers

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # Replace list comprehension with explicit loop
            result = []
            if hidden_states is not None:
                result.append(hidden_states)
            if all_hidden_states is not None:
                result.append(all_hidden_states)
            if all_self_attentions is not None:
                result.append(all_self_attentions)
            return tuple(result)
            
        # Here the original code used a BaseModelOutput
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
    # Apply the patch to encoder's forward method
    model.encoder.forward = types.MethodType(patched_encoder_forward, model.encoder)
    print("Successfully patched encoder forward method")

# Save the original configuration files for reference
print("Saving original model configuration files...")
# Save the model config
model.config.to_json_file(os.path.join(OUTPUT_DIR, "config.json"))
# Save the processor config
processor.feature_extractor.to_json_file(os.path.join(OUTPUT_DIR, "preprocessor_config.json"))

# Extract key parameters from configurations
model_hidden_size = model.config.hidden_size  # Should be 768 based on the config
sample_rate = processor.feature_extractor.sampling_rate  # Should be 16000

print(f"Model configuration: hidden_size={model_hidden_size}, sample_rate={sample_rate}")

# Create a wrapper class that combines preprocessing and model inference
class Music2VecModelWrapper(torch.nn.Module):
    def __init__(self, processor, model, config_sample_rate):
        super().__init__()
        self.processor = processor
        self.model = model
        # Save embedding dimension for reference
        self.embedding_dim = model.config.hidden_size
        # Use the sample rate from config
        self.sample_rate = config_sample_rate
    
    def preprocess_audio(self, audio_data):
        """Manually preprocess audio data without using the processor API directly"""
        # Ensure audio_data is 2D [batch_size, sequence_length]
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)  # Add batch dimension
            
        # Create attention mask (all 1s for the actual audio)
        attention_mask = torch.ones_like(audio_data, dtype=torch.long)
        
        # Return as a dict similar to processor output but with tensors directly
        return {
            "input_values": audio_data,
            "attention_mask": attention_mask
        }
    
    def forward(self, audio_data):
        """
        Process audio data and extract embeddings.
        
        Args:
            audio_data: Raw audio waveform tensor [batch_size, audio_length] or [audio_length]
            
        Returns:
            Normalized embedding vector [batch_size, embedding_dim]
        """
        # Process the audio data
        with torch.no_grad():
            # Ensure input is 2D [batch_size, sequence_length]
            if audio_data.dim() == 1:
                audio_data = audio_data.unsqueeze(0)  # Add batch dimension
                
            # Manual preprocessing to avoid TracerWarning issues with processor
            inputs = self.preprocess_audio(audio_data)
            
            # Extract tensors from inputs dictionary
            input_values = inputs["input_values"]
            attention_mask = inputs["attention_mask"]
            
            # Get Model Output
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True  # Force return_dict=True for consistent structure
            )
            
            # Get the last hidden state
            last_hidden = outputs.last_hidden_state
            
            # Average pooling over the time dimension to get a fixed-size embedding
            embedding = last_hidden.mean(dim=1)  # [batch_size, hidden_size]
            
            # Normalize the embedding
            norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
            normalized_embedding = embedding / (norm + 1e-8)
            
            return normalized_embedding

# Create an instance of the wrapper
print("Creating model wrapper...")
wrapped_model = Music2VecModelWrapper(processor, model, sample_rate)

# Create example input for tracing
# Generate 1 second of silence at 16kHz (shape must be [batch_size, audio_length])
print("Creating example input...")
example_input = torch.zeros(1, 16000, dtype=torch.float32)  # [batch_size, audio_length]

# Export the model using scripting instead of tracing
print("Scripting model...")
try:
    # Use torch.jit.script to create a TorchScript model
    scripted_model = torch.jit.script(wrapped_model)
    
    # Save metadata about the model
    model_info = {
        "embedding_dim": wrapped_model.embedding_dim,
        "sample_rate": sample_rate,
        "model_type": "music2vec-v1",
        "context_window": 30,  # 30 seconds as per the original model spec
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_hidden_layers": model.config.num_hidden_layers,
        "model_format": "scripted"
    }
    
    # Save the scripted model
    model_path = os.path.join(OUTPUT_DIR, "pytorch_model.pt")
    print(f"Saving scripted model to {model_path}")
    scripted_model.save(model_path)
    
    # Save model info for reference
    with open(os.path.join(OUTPUT_DIR, "model_info.txt"), "w") as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
    
    # Create a README file with information about the model
    readme_content = """# Music2Vec Scripted Model

This directory contains a TorchScript scripted version of the music2vec-v1 model.

## Model Details
- Original model: music2vec-v1 (m-a-p/music2vec-v1 on Hugging Face)
- Model architecture: Data2VecAudio
- Embedding dimension: {0}
- Context window: 30 seconds
- Sample rate: {1} Hz
- Number of layers: {2}
- Number of attention heads: {3}
- Hidden size: {4}

## Original Configuration
The original model and processor configuration files are included:
- config.json: Model architecture configuration
- preprocessor_config.json: Audio preprocessing configuration

## Usage in C++
This model can be loaded directly using LibTorch's torch::jit::load() function.
The model takes raw audio waveform data and returns a normalized embedding vector.

Input:
- Raw audio as float tensor [batch_size, audio_length]

Output:
- Normalized embedding vector [batch_size, {0}]

Created on: {5}
"""
    
    # Get current date and time
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write README file
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write(readme_content.format(
            wrapped_model.embedding_dim,
            sample_rate,
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            model.config.hidden_size,
            current_time
        ))
    
    print("Conversion completed successfully!")
    print(f"Model saved to: {os.path.abspath(OUTPUT_DIR)}")
    
except Exception as script_error:
    print(f"Scripting failed: {script_error}")
    
    # Try exporting model with just basic operations
    print("Attempting to create simplified export...")
    try:
        # Create a simpler wrapper that just runs inference without complex preprocessing
        class SimpleMusic2VecWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model
                self.embedding_dim = base_model.config.hidden_size
                
            def forward(self, input_values, attention_mask=None):
                """Simple forward pass without complex preprocessing"""
                with torch.no_grad():
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_values, dtype=torch.long)
                        
                    # Run model forward with return_dict=True to avoid list comprehension
                    outputs = self.model(
                        input_values=input_values,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    
                    # Get embeddings by mean pooling
                    last_hidden = outputs.last_hidden_state
                    embedding = last_hidden.mean(dim=1)
                    
                    # Normalize
                    norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
                    return embedding / (norm + 1e-8)
        
        # Create the simplified wrapper
        simple_wrapper = SimpleMusic2VecWrapper(model)
        
        # Create proper example input for the simplified model
        example_inputs = (example_input, torch.ones(1, 16000, dtype=torch.long))
        
        # Try tracing the simplified model
        traced_simple_model = torch.jit.trace(simple_wrapper, example_inputs)
        
        # Save the traced model
        simplified_model_path = os.path.join(OUTPUT_DIR, "pytorch_model_simplified.pt")
        print(f"Saving simplified model to {simplified_model_path}")
        traced_simple_model.save(simplified_model_path)
        
        # Save a note about this being a simplified version
        with open(os.path.join(OUTPUT_DIR, "simplified_model_info.txt"), "w") as f:
            f.write("This is a simplified version of the model that requires preprocessing outside of TorchScript.\n")
            f.write(f"Input: preprocessed audio in shape [batch_size, sequence_length]\n")
            f.write(f"Output: normalized embeddings of size [batch_size, {simple_wrapper.embedding_dim}]\n")
            
        print("Saved simplified model as a fallback.")
        
    except Exception as simplified_error:
        print(f"Even simplified export failed: {simplified_error}")
        print("Could not convert model for C++ usage.")
        raise  # Re-raise the error 