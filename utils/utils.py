import cv2

def save_tensor(tensor, output_path):
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    cv2.imwrite(output_path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))