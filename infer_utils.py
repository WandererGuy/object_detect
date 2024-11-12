import cv2
import os.path as osp
from tqdm import tqdm
import os 
import time 


def fix_path(path):
    path = str(path)
    new_path = path.replace('\\\\','/') 
    return new_path.replace('\\','/')


enable_decorator = True
def timeit(method):
    def timed(*args, **kw):
        if enable_decorator:
            ts = time.perf_counter()  # Record start time
            result = method(*args, **kw)  # Call the original function
            te = time.perf_counter()  # Record end time
            print(f"'{method.__name__}'{te - ts:.2f} sec")
        else:
            result = method(*args, **kw)
        return result  # Return the result of the original function
    return timed

def frame_crop(video_path, save_folder, freq=4):
    cap = cv2.VideoCapture(video_path)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    # fps = cap.get(cv2.CAP_PROP_FPS)
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id % freq == 0:
                    frame = remove_padding(frame)
                    save_name = "{:03d}".format(frame_id)+".png"
                    path_save = osp.join(save_folder, save_name)
                    cv2.imwrite(path_save, frame)
        frame_id = frame_id + 1

def frame_crop_folder(folder_path, save_folder, freq=4):
    frame_id = 0
    for filename in tqdm(os.listdir(folder_path), total=len(os.listdir(folder_path))):
        frame = cv2.imread(osp.join(folder_path, filename))
        if frame_id % freq == 0:
                frame = remove_padding(frame)
                save_name = "{:03d}".format(frame_id)+".png"
                path_save = osp.join(save_folder, save_name)
                cv2.imwrite(path_save, frame)
        frame_id = frame_id + 1

def remove_padding(image):
    # image = cv2.imread(source_path)

    # Convert to grayscale (black padding is easy to detect in grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary mask (black padding is 0, other regions are 255)
    '''
    Any pixel value greater than 1 will be converted to 255 (white).
    Any pixel value less than or equal to 1 will be set to 0 (black).

    '''
    _, thresh = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)

    # Find contours of the white regions in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Finds the four vertices of a straight rect. Useful to draw the rotated rectangle.
    x, y, w, h = cv2.boundingRect(max_contour)  # Get bounding box of the contour (minimum rect)


    '''
    x1 ----> x2
    |      |
    |      |
    |      |
    |      v
    x4 <---- x3
    '''
    cropped_image = image[int(y):int(y+h), int(x):int(x+w)]
    
    return cropped_image
