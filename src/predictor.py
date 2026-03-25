import torch
import numpy as np
import sys
import os

# from pretraitement import preprocess_image

# This adds the root directory to the python path so it can find the 'MNIST' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MNIST.MnistCNN import MnistCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MnistCNN(num_classes=10, dropout=0.3).to(device)



class Predictor:
    def __init__(self, model_path , energy_threshold=-5.0):

        self.model = MnistCNN(num_classes=10, dropout=0.3).to(device)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.energy_threshold = energy_threshold
        
        self.model.eval()

    def predict(self, image ):
        
        image_tensor = torch.from_numpy(image).float()
        
        

        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.nn.functional.softmax(output, dim=1)

            print("Prediction probabilities:", output.cpu().numpy())

            energy = -torch.logsumexp(output, dim=1)
            print("Energy:", energy.item())

            if energy.item() > self.energy_threshold:
                print("Energy above threshold, rejecting prediction.")
                return None, None
            
            conf, predicted_class = torch.max(prediction, 1)
        return predicted_class.item(), conf.item()

    
# pred = Predictor("../models/best_model_v2.pth")
# image = preprocess_image("../image.png")
# pred.predict(image)