import os
import argparse
import subprocess
import time

import cv2


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--image_path', type=str,
                      required=False, default='./tmp.png',
                      help='temporary captured image path')
    args.add_argument('-o', '--output_path', type=str,
                      required=False, default='./face.png',
                      help='output face image path')
    return args.parse_args()


def run_create(image_path, output_path):
    assert isinstance(image_path, str)
    assert isinstance(output_path, str)
    cmd = f'./create_new_face {image_path} {output_path}'
    res = subprocess.run(cmd, shell=True)
    return res.returncode == 0


def main(video_path=None):
    has_video = video_path is not None
    args = get_args()
    image_path = args.image_path
    output_path = args.output_path
    if has_video:
        print('displaying video, press q to capture and stop')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        last_frame_time = 0
    else:
        print('displaying webcam, press q to capture and stop')
        cap = cv2.VideoCapture(0)
        last_frame_time = 0
        fps = 0
    if not cap.isOpened():
        raise RuntimeError('Failed to open camera/video')
    frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if has_video:
                cap.release()
                cap = cv2.VideoCapture(video_path)
            continue
        if not has_video:
            frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        if has_video:
            cur_time = time.time()
            if cur_time - last_frame_time < 1 / fps:
                time.sleep(1 / fps - (cur_time - last_frame_time) - 0.01)
            last_frame_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if frame is None:
        raise RuntimeError('Failed to capture image')

    cv2.imshow('captured', frame)
    print('press "c" to confirm, other key to quit')
    if cv2.waitKey(0) & 0xFF == ord('c'):
        cv2.imwrite(image_path, frame)
        if run_create(image_path, output_path):
            print(f'face created at {output_path}')
        else:
            print('failed to create face')
    else:
        print('canceled')


if __name__ == '__main__':
    main()
