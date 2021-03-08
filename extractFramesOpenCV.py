import os
import cv2

phases = ['Train', 'Test', 'Validation']
for phase in phases:
    path = os.path.join('/path/to/video/files/', phase)
    subjects = os.listdir(path)

    for subject in subjects:
        print(phase, subject, flush=True)
        videos = os.listdir(os.path.join(path, subject))
        for video in videos:
            videoPath = os.path.join(path, subject, video, os.listdir(os.path.join(path, subject, video))[0])
            videoPathFrames = '/'.join(videoPath.split('.')[0].split('/')[:-1]).replace(phase, phase + 'Frames')
            os.makedirs(videoPathFrames, exist_ok=True)

            capture = cv2.VideoCapture(videoPath)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            count = 0
            i = 0
            retaining = True
            while count < frame_count and retaining:
                retaining, frame = capture.read()
                if frame is None:
                    continue
                cv2.imwrite(filename=os.path.join(videoPathFrames, '{}.jpg'.format(str(i))), img=frame)
                i += 1
                count += 1
            capture.release()
