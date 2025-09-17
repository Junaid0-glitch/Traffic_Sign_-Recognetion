# Traffic Sign Classification

This project is a Traffic Sign Classification web app built with Streamlit and TensorFlow. It uses a pre-trained CNN model to predict the class of a traffic sign from an uploaded image.

## Features
- Upload an image of a traffic sign
- Get instant prediction of the traffic sign class
- Uses a trained model (`Traffic_signal_model.h5`) for inference

## Files
- `app.py`: Streamlit web app for traffic sign classification
- `Traffic_signal_model.h5`: Pre-trained CNN model
- `Traffic Signal Classification.ipynb`: Jupyter notebook for model training and evaluation
- `requirements.txt`: Python dependencies

## Getting Started
1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```powershell
   streamlit run app.py
   ```
3. Upload a traffic sign image and view the prediction.

## Model Training
See `Traffic Signal Classification.ipynb` for details on data preprocessing, model architecture, training, and evaluation.

## License
This project is for educational purposes.
