# Fine-Tuning Language Models with LoRA on Kaggle

This repository contains a Jupyter notebook for fine-tuning language models using LoRA (Low-Rank Adaptation) and 4-bit quantization techniques optimized for Kaggle's free tier GPU environment. The implementation focuses on memory-efficient training while maintaining model performance.

## Features
- **4-bit Quantization**: Uses BitsAndBytes for memory-efficient model loading
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning with PEFT library
- **Kaggle Optimized**: Training arguments optimized for Kaggle's free tier (T4 GPU)
- **Memory Efficient**: Gradient accumulation and batch size optimization
- **Clean Implementation**: Well-documented notebook with step-by-step process

## Requirements
- **Python 3.8+**
- **CUDA-compatible GPU** (T4 recommended, as available on Kaggle)
- All dependencies listed in `requirements.txt`

## Kaggle Usage
1. **Upload to Kaggle**:
   - Upload `FineTunning.ipynb` to a new Kaggle notebook
   - Or fork this repository directly in Kaggle

2. **Setup Dataset**:
   - Upload your training dataset to Kaggle Datasets
   - Update the dataset path in the first cell:
   ```python
   with open("/kaggle/input/YOUR-DATASET-NAME/your-file.json", "r") as f:
   ```

3. **Configure Model**:
   - Replace `"YOUR MODEL ID"` with your desired model (e.g., `"microsoft/DialoGPT-medium"`)
   - Add your Hugging Face token:
   ```python
   login(token="your_actual_huggingface_token")
   ```

4. **Run the Notebook**:
   - Execute cells sequentially
   - Training will take ~30-40 minutes on Kaggle's free tier
   - Model will be saved as LoRA adapters for lightweight storage

## Local Usage (Alternative)
1. Clone the repository:
   ```bash
   git clone https://github.com/Abhijeet-ist/FineTunning.git
   cd FineTunning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook FineTunning.ipynb
   ```

## Training Configuration
The notebook is optimized for Kaggle's constraints:
- **Batch size**: 1 with gradient accumulation steps of 4
- **Max steps**: 500 (prevents timeout)
- **Mixed precision**: FP16 enabled for memory savings
- **Model saving**: Only LoRA adapters saved (lightweight)

## Dataset Format
Your dataset should be in JSON format with instruction-response pairs:
```json
[
  {
    "instruction": "Your training instruction here",
    "response": "Expected model response here"
  },
  ...
]
```

## Key Libraries Used
- **transformers**: Model loading and training
- **peft**: LoRA implementation
- **bitsandbytes**: 4-bit quantization
- **datasets**: Data handling and preprocessing
- **accelerate**: Training optimization

## Tips for Kaggle
- Enable GPU in notebook settings
- Use datasets feature to upload training data
- Monitor GPU memory usage during training
- Save outputs regularly to prevent data loss

## Contributing
Feel free to submit issues and enhancement requests!

