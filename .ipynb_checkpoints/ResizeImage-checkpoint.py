import WaveletTransformation as wt
import numpy as np
import cv2
class_dict = {
    'Lionel Messi': 0,
    'Maria Sharapova': 1,
    'Roger Federer': 2,
    'Serena Williams': 3,
    'Virat Kohli': 4
    }
# Feature extractor for a single image
def resize_image_trained(cropped_image_path, label):
    Features, Target = [], []
    
    raw_image = cv2.imread(cropped_image_path)
    if raw_image is None:   # in case file is missing or unreadable
        return None, None
    
    scaled_raw_image = cv2.resize(raw_image, (32, 32))
    
    har_image = wt.w2d(raw_image, 'db1', 5)
    scaled_har_image = cv2.resize(har_image, (32, 32))
    
    combined_image = np.hstack((
        scaled_raw_image.reshape(32 * 32 * 3),
        scaled_har_image.reshape(32 * 32)
    ))
    
    Features.append(combined_image)
    Target.append(label)
    
    Features = np.array(Features).astype(float)
    Target = np.array(Target)
    
    return Features, Target

def resize_image(cropped_image_path):
    Features= []
    
    raw_image = cv2.imread(cropped_image_path)
    if raw_image is None:   # in case file is missing or unreadable
        return None, None
    
    scaled_raw_image = cv2.resize(raw_image, (32, 32))
    
    har_image = wt.w2d(raw_image, 'db1', 5)
    scaled_har_image = cv2.resize(har_image, (32, 32))
    
    combined_image = np.hstack((
        scaled_raw_image.reshape(32 * 32 * 3),
        scaled_har_image.reshape(32 * 32)
    ))
    
    Features.append(combined_image)
    
    Features = np.array(Features).astype(float)
    return Features

Feature , target = resize_image_trained('./Celebrity/cropped/Virat Kohli/Virat Kohli39.png' , 'Virat Kohli')
print(target)