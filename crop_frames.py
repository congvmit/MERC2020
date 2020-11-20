import cv2
import numpy as np
from mtcnn.detector import detect_faces, get_net
from mtcnn.visualization_utils import show_bboxes
from align_faces import warp_and_crop_face
import mipkit
import os
import glob
import traceback
from tqdm import tqdm


def crop_faces(args):
    video_paths = args['video_paths']
    save_dir = args['save_dir']
    skip = args['skip']
    crop_size = args['crop_size']
    log_file = args['log_file']
    tqdm_pos = args['tqdm_pos']

    # init model
    net = get_net()
    pbar = tqdm(total=len(video_paths), position=tqdm_pos)

    for video_path in tqdm(video_paths):
        pbar.update(1)
        error = False
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(video_path).split('.')[0]
        ext = os.path.basename(video_path).split('.')[1]

        if total_frames < 150:
            skip = 5

        if total_frames < 120:
            skip = 4

        if total_frames < 100:
            skip = 3

        for i in range(total_frames):
            try:
                if i % skip == 0:
                    # Write to file
                    ret, frame = cap.read()
                    cap.set(1, i+1)
                    bounding_boxes, landmarks = detect_faces(frame,
                                                             min_face_size=50.0,
                                                             thresholds=[
                                                                 0.5, 0.6, 0.7],
                                                             nms_thresholds=[
                                                                 0.6, 0.6, 0.6],
                                                             net=net)
                    for lm in landmarks:
                        crop_face = warp_and_crop_face(frame, np.array(lm).reshape([2, 5]),
                                                       crop_size=crop_size,
                                                       default_square=True,
                                                       outer_padding=(0, 0))

                        folder_to_save = os.path.join(
                            save_dir, basename + '_' + str(total_frames))
                        os.makedirs(folder_to_save, exist_ok=True)
                        path_to_save = os.path.join(*[folder_to_save,
                                                      basename + '_' + mipkit.convert_int_to_str(i) + '.jpg'])
                        cv2.imwrite(path_to_save, crop_face)
            except Exception as e:
                print('Error while reading ' + video_path +
                      '--at: ' + mipkit.convert_int_to_str(i))
                print(traceback.format_exc())
                error = True
                break

        if not error:
            with open(log_file, 'a') as f:
                f.write(video_path + '\n')


# Load training dataset
log_file = 'train_processed_video.txt'
load_dir = '/mnt/hdd4T/MERC2020/2020-1/train/*.mp4'
save_dir = '/mnt/hdd4T/MERC2020/2020-1/processed_data/train'
skip = 6
crop_size = 128, 128

# ================================================
video_file_path = sorted(glob.glob(load_dir))

if not os.path.isfile(log_file):
    with open(log_file, 'w') as f:
        pass

with open(log_file, 'r') as f:
    processed_paths = f.readlines()

processed_paths = set(map(lambda x: x.strip().replace(
    '//', '/').replace('\n', ''), processed_paths))

print('Loaded', len(video_file_path))
video_file_path = list(set(video_file_path).difference(processed_paths))
print("Processing ", len(video_file_path), 'paths')

splitted_paths = mipkit.split_seq(video_file_path)

inputs = [{'video_paths': video_paths,
           'save_dir': save_dir,
           'skip': skip,
           'crop_size': crop_size,
           'log_file': log_file,
           'tqdm_pos': i} for i, video_paths in enumerate(splitted_paths)]

mipkit.multiprocessing.pool_worker(target=crop_faces,
                                   inputs=inputs,
                                   num_worker=8)
