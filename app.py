import torch
import boto3

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[-1, :, :])  # Get the last output
        return out

def load_model_from_s3():
    print("Starting model download from S3...")
    s3 = boto3.client('s3', region_name='eu-north-1')
    bucket_name = 'my-lstm-models'
    model_key = 'best_model.pth'
    local_path = 'best_model.pth'
    
    try:
        s3.download_file(bucket_name, model_key, local_path)
        print(f"Model downloaded to {local_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

    # Define the model architecture
    input_size = 5  # Adjust based on your model's expected input
    hidden_size = 64  # Adjust as needed
    output_size = 1  # Adjust based on your model's output
    model = LSTMModel(input_size, hidden_size, output_size)

    try:
        # Use strict=False to ignore extra keys in the state dict
        model.load_state_dict(torch.load(local_path, map_location=torch.device('cpu')), strict=False)
        print("Model loaded successfully with some keys ignored (if any).")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model.eval()  # Set model to evaluation mode
    return model


def run_inference(model):
    # For demonstration, generate a random input tensor
    sample_input = torch.randn(10, 1, 5)  # Batch size of 1, sequence length of 10, input size of 5
    print(f"Running inference with input shape: {sample_input.shape}")
    
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Inference result: {output}")
    return output

if __name__ == "__main__":
    model = load_model_from_s3()
    if model:
        output = run_inference(model)
