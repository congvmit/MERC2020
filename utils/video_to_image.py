import cv2
import csv
import os
import glob
import threading
import mipkit

def video_to_frame(inputs):
    video_path, save_dir = inputs
    cap = cv2.VideoCapture(video_path)
    name = video_path.split('/')[-1].split('.')[0]
    save_folder_path = save_dir + '/'+ str(name[:5])
    
    if not (os.path.isdir(save_folder_path)):
        os.makedirs(os.path.join(save_folder_path))
    count = 0
    fps = 15
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if(count%fps==0):
                save_file_path =  save_folder_path +"/" + name + "_{:d}.jpg".format(count)
                cv2.imwrite(os.path.join(save_file_path), frame)
            count += 1
        else:
            break

if __name__ == "__main__":

    load_dir = '/mnt/DATA2/congvm/MERC2020/2020-1/train/*.mp4'
    save_dir = "/mnt/DATA2/congvm/MERC2020/2020-1/image_frame_train_1/"
    
    video_file_path = sorted(glob.glob(load_dir))
    
    inputs = [[fpath, save_dir] for fpath in video_file_path]
    mipkit.multiprocessing.pool_worker(inputs=inputs, target=video_to_frame, num_worker=6)
    print('Done')
    
