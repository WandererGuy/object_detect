print("............. Initialization .............")
import ultralytics
from ultralytics import YOLO
import requests
from infer_utils import frame_crop, frame_crop_folder, timeit, fix_path
import os 
import uuid
import logging
import cv2
from tqdm import tqdm
import threading 
from queue import Queue
import numpy as np
import json
import gc
import time 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",  # Customized format
    datefmt="%Y-%m-%d %H:%M:%S"  # Timestamp format
)

model = YOLO("yolo11x-seg.pt")
# Assuming `model` is your PyTorch model
model.to('cuda')
print('model on gpu' ,next(model.parameters()).is_cuda)
current_script_directory = os.path.dirname(os.path.abspath(__file__))
FRAME_FOLDER = os.path.join(current_script_directory, "frames")
OBJECT_FOLDER = os.path.join(current_script_directory,"video_objects")
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(OBJECT_FOLDER, exist_ok=True)
LABELS = model.names
MIN_CONF = 0.45
WINDOW_AUTO_SAVE_EXTENSION = '.jpg'
BATCH_SIZE = 8 # batch 16 take 4.5 gb vram and 151s compare to batch 8 TAKE 2.5 gb only 147 sec 



def saving_object_images(video_name, frame_name):
    t = [OBJECT_FOLDER, video_name, frame_name]
    sub_image_path = os.path.join(*t)
    image_name = str(uuid.uuid4())
    return sub_image_path, image_name


def generate_folder_frame(mode, video_path, freq):
    uuid_str = str(uuid.uuid4())
    folder_frame = os.path.join(FRAME_FOLDER, uuid_str)
    os.makedirs(folder_frame, exist_ok=True)
    if mode == 'video':
        frame_crop(video_path, folder_frame, freq=freq)
    if mode == 'folder':
        frame_crop_folder(video_path, folder_frame, freq=freq)
    return folder_frame


@timeit
def process_frames(folder_frame):
    
    '''
    arg:
        folder_frame: folder contain frames (already removed existing padding)
    return:
        object image path, box_coordinates, object_label
    func:
        detect object in all frames and save them in structure
        ___video
        ______frame
        _________class
        ____________image_object
    '''
    k = {}
    crop_isolated_object_path_ls = []
    logging.info ('---- DETECTING OBJECTS IN FRAMES ----')
    video_name = str(uuid.uuid4())

    logging.info ('processing number of frames: ' + str(len(os.listdir(folder_frame))))
    logging.info ('done all object detection in all frames')
    logging.info (f'start save objects in folder: {os.path.join(OBJECT_FOLDER, video_name)}')

    count = 0
    for single_frame_result in model.predict(
    source = folder_frame,
    batch=BATCH_SIZE,
    conf=MIN_CONF,
    imgsz=640,
    verbose=False,
    save=False,
    stream=True
):  
        count += 1
        logging.info ('prediction frame: ' + str(count))
        for item in single_frame_result:
                ori_img = item.orig_img
                box = item.boxes
                box_coordinates = box.xyxy.tolist()[0] # top-left , bottom-right
                object_label =  LABELS[(box.cls).item()]
                frame_name = os.path.basename(item.path.split(".")[0])
                sub_image_path, image_name = saving_object_images(
                                    video_name = str(video_name), 
                                    frame_name = str(frame_name)
                                    )
                item.save_crop(sub_image_path, file_name = image_name) # ultralytics save crop 
                crop_object_path = os.path.join(sub_image_path, object_label, image_name) + WINDOW_AUTO_SAVE_EXTENSION
                full_path_object = os.path.join(sub_image_path, object_label, image_name) + WINDOW_AUTO_SAVE_EXTENSION
                mask = item.masks
                # isolate object + crop according to detection box
                crop_isolated_object_path = save_isolated_object(mask, full_path_object, ori_img, box_coordinates)
                crop_isolated_object_path_ls.append(crop_isolated_object_path)
                # k[crop_isolated_object_path] = crop_object_path
                k[crop_isolated_object_path] = crop_isolated_object_path

    logging.info (f'DONE save objects in folder: {os.path.join(OBJECT_FOLDER, video_name)}')
    return k, crop_isolated_object_path_ls

def send_server(url, payload, output_queue):
    files=[]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    output_queue.put(response.json())


def fix_background(b_mask, img):
      # OPTION-1: Isolate object with black background
  # if save_path.endswith('.png'):
  #       transparent = True
  # else:
  #       transparent = False
  transparent = False
  if not transparent:
    # Create 3-channel mask
    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    # Isolate object with binary mask
    isolated = cv2.bitwise_and(mask3ch, img)
  else:
    # OPTION-2: Isolate object with transparent background (when saved as PNG)
    isolated = np.dstack([img, b_mask])
  return isolated

def mask_img(img, mask):
  # Create binary mask
  b_mask = np.zeros(img.shape[:2], np.uint8)
  #  Extract contour result
  contour = mask.xy.pop()
  #  Changing the type
  contour = contour.astype(np.int32)
  #  Reshaping
  contour = contour.reshape(-1, 1, 2)
  # Draw contour onto mask
  _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
  isolated = fix_background(b_mask, img)
  return isolated

def save_isolated_object(mask, full_path_object, img, box_coordinates):
    isolated_object = mask_img(img, mask)
    modify_part  =  '.' + full_path_object.split('.')[-1] 
    full_path_crop_isolated_object = full_path_object.replace(modify_part, "_mask" + modify_part)
    # crop isolated accỏiding to detection box
    x1, y1, x2, y2 = box_coordinates
    crop_isolated_object = isolated_object[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite(full_path_crop_isolated_object, crop_isolated_object)
    return full_path_crop_isolated_object

# def save_wb_object(mask, full_path_object, wb_frame_path):
#     img = cv2.imread(wb_frame_path)
#     isolated_wb_object = mask_img(img, mask)
#     modify_part  =  '.' + full_path_object.split('.')[-1] 
#     full_path_wb_object = full_path_object.replace(modify_part, "_mask" + modify_part)
#     cv2.imwrite(full_path_wb_object, isolated_wb_object)

def main():
    """
    # step 1 generate folder of frame
    mode = folder 
    assume video has been split into frames
    let 
        freq = 1 
        video_path = path_to_folder 
    
    mode = video
    let 
        freq = 4
        video_path = path_to_video

    step 2 
        detect object in all frames, 
        isolated them 

    """
    freq=1
    mode = 'folder' # or mode = 'folder' for folder, video_path become folder path
    video_path = r"C:\Users\Admin\CODE\work\OBJECT_COLOR\object_detect\frames\4bf459a7-b37e-4d0a-8b47-3afcf8c493c0"
    logging.info (f'generate folder of frame from video')
    folder_frame = generate_folder_frame(mode, video_path, freq=freq)
    logging.info (f'save every {freq} frames to folder{folder_frame}')
    logging.info ('start detect and saving object from images as background task')
    logging.info ('Also, send folder_frame path to wb server')
    res, crop_isolated_object_path_ls = process_frames(folder_frame)


    print ('-----------------------------------')
    print ('DONEEEEE')



if __name__ == "__main__":
    main()


