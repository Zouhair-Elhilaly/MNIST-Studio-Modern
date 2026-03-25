from PIL import Image, ImageOps
import numpy as np

# def preprocess_image(img) -> np.ndarray:
#     if isinstance(img, str):
#         img = Image.open(img)
    
#     img = img.convert("L")
    
#     img = ImageOps.invert(img)
    
#     arr = np.array(img)
    
#     coords = np.argwhere(arr > 20)
#     if coords.size == 0:
#         arr = np.zeros((28, 28), dtype=np.float32)
#         return arr.reshape(1, 28, 28, 1)
    
#     y0, x0 = coords.min(axis=0)
#     y1, x1 = coords.max(axis=0) + 1
#     arr = arr[y0:y1, x0:x1]
    
#     img = Image.fromarray(arr)
#     img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    
#     new_img = Image.new("L", (28, 28), 0)
#     x = (28 - img.width) // 2
#     y = (28 - img.height) // 2
#     new_img.paste(img, (x, y))
    

#     arr = np.array(new_img).astype("float32") / 255.0
#     arr = arr.reshape(1, 28, 28, 1)
    
#     return arr



def preprocess_image(img) -> np.ndarray:
    """
    Prepares the input image from the Streamlit canvas for the PyTorch CNN model.
    Steps: Grayscale conversion, Inversion, Centering, Normalization, and Reshaping.
    """
    
    # 1. Convert input to PIL Image if it's a NumPy array (from Streamlit Canvas)
    if isinstance(img, np.ndarray):
        # Canvas output is typically RGBA; we convert to RGB first
        img = Image.fromarray(img.astype('uint8'), 'L')
    elif isinstance(img, str):
        # If a file path is provided instead
        img = Image.open(img)

    # 2. Convert to Grayscale ('L' mode)
    img = img.convert("L")
    
    # 3. Invert colors: MNIST models expect white digits (255) on black background (0)
    # Streamlit canvas usually provides black drawing on white background
    img = ImageOps.invert(img)
    
    # 4. Find the bounding box of the digit to crop extra empty space
    arr = np.array(img)
    coords = np.argwhere(arr > 20) # Find pixels brighter than threshold
    
    if coords.size == 0:
        # If the canvas is empty, return a blank normalized tensor
        blank = np.zeros((28, 28), dtype=np.float32)
        blank = (blank - 0.1307) / 0.3081
        return blank.reshape(1, 1, 28, 28)
    
    # Get the min/max coordinates for cropping
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    digit_crop = arr[y0:y1, x0:x1]
    
    # 5. Resize the digit while maintaining aspect ratio (to fit roughly 20x20)
    img_digit = Image.fromarray(digit_crop)
    img_digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    
    # 6. Create a new 28x28 black canvas and paste the resized digit in the center
    final_img = Image.new("L", (28, 28), 0)
    offset_x = (28 - img_digit.width) // 2
    offset_y = (28 - img_digit.height) // 2
    final_img.paste(img_digit, (offset_x, offset_y))
    
    # 7. Convert to NumPy array and Normalize to [0, 1] range (equivalent to transforms.ToTensor())
    final_arr = np.array(final_img).astype("float32") / 255.0
    
    # 8. Apply specific MNIST Normalization used during training (mean=0.1307, std=0.3081)
    # This ensures the input distribution matches the training data
    mean = 0.1307
    std = 0.3081
    final_arr = (final_arr - mean) / std
    
    # 9. Reshape to match PyTorch input: (Batch_Size, Channels, Height, Width)
    # Since it's a single grayscale image: (1, 1, 28, 28)
    return final_arr.reshape(1, 1, 28, 28)