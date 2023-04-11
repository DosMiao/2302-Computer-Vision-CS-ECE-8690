import os
import cv2
import numpy as np


class BGSubModel:
    """Background subtraction model.
    """

    def __init__(self, first_frame, alpha, tm):
        self.mean = np.float32(first_frame)
        self.var = np.ones_like(self.mean) * 100
        self.alpha = alpha
        self.tm = tm

    def classify(self, current_frame):
        diff = np.abs(np.float32(current_frame) - self.mean)
        fg_mask = np.where(
            diff > (self.tm * np.sqrt(self.var)), 255, 0).astype(np.uint8)
        return fg_mask

    def update(self, current_frame):
        alpha_mask = np.where(np.abs(np.float32(current_frame) - self.mean)
                              <= (self.tm * np.sqrt(self.var)), self.alpha, 0).astype(np.float32)
        self.mean = (1 - alpha_mask) * self.mean + \
            alpha_mask * np.float32(current_frame)
        self.var = (1 - alpha_mask) * self.var + alpha_mask * \
            (np.float32(current_frame) - self.mean) ** 2


# Parameters
ALPHA = 0.001
TM = 2
INPUT_PATH = './CV2023_HW4B/input'
OUTPUT_PATH = './CV2023_HW4B/output'


def main():
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    flist = [f for f in os.listdir(INPUT_PATH) if f.endswith('.png')]
    flist = sorted(flist)
    n = len(flist)

    # Read the first image and initialize the model
    im = cv2.imread(os.path.join(INPUT_PATH, flist[0]))
    bg_model = BGSubModel(im, ALPHA, TM)

    # Set up the VideoWriter objects
    # the name should contain the parameters of the model, ALPHA and TM
    video_file = os.path.join(OUTPUT_PATH, f'op_{ALPHA}_{TM}.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        video_file, fourcc, 10, (im.shape[1], im.shape[0]))

    fgmask_video_file = os.path.join(OUTPUT_PATH, f'op_fg_{ALPHA}_{TM}.avi')
    fgmask_video_writer = cv2.VideoWriter(
        fgmask_video_file, fourcc, 10, (im.shape[1], im.shape[0]))

    # Main loop
    for fr in range(n):
        im = cv2.imread(os.path.join(INPUT_PATH, flist[fr]))
        fg_mask = bg_model.classify(im)
        bg_model.update(im)

        # Save the results for specific frames
        if fr in [5, 100, 400]:
            fname = f'FGmask_{ALPHA}_{TM}_{flist[fr]}'
            fname_wpath = os.path.join(OUTPUT_PATH, fname)
            cv2.imwrite(fname_wpath, fg_mask)

            fname = f'BGmean_{ALPHA}_{TM}_{flist[fr]}'
            fname_wpath = os.path.join(OUTPUT_PATH, fname)
            cv2.imwrite(fname_wpath, bg_model.mean.astype('uint8'))

        # Write the current frame and FGmask to the videos
        video_writer.write(im)
        fgmask_video_writer.write(fg_mask)

    video_writer.release()
    fgmask_video_writer.release()


if __name__ == '__main__':
    main()
