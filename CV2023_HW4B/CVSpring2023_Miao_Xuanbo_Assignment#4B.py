# %% Import Packages
import os
import cv2
import numpy as np

# %% Background Subtraction Model


class BGSubModel:

    # The model is initialized with the first frame.
    def __init__(self, first_frame, alpha, tm):
        self.mean = np.float32(first_frame)
        self.var = np.ones_like(self.mean) * 100
        self.alpha = alpha
        self.tm = tm

    # Classify the current frame as foreground or background
    def classify(self, current_frame):
        diff = np.abs(np.float32(current_frame) - self.mean)
        fg_mask = np.where(
            diff > (self.tm * np.sqrt(self.var)), 255, 0).astype(np.uint8)
        return fg_mask

    # Update the model with the current frame
    def update(self, current_frame):
        alpha_mask = np.where(np.abs(np.float32(current_frame) - self.mean)
                              <= (self.tm * np.sqrt(self.var)), self.alpha, 0).astype(np.float32)
        inv_alpha_mask = 1 - alpha_mask
        self.mean = inv_alpha_mask * self.mean + \
            alpha_mask * np.float32(current_frame)
        self.var = inv_alpha_mask * self.var + \
            alpha_mask * (np.float32(current_frame) - self.mean) ** 2


# %% Main
# Parameters
ALPHA_list = [0.001]
TM_list = [2]

# Files & Folders
INPUT_PATH = 'c:/Users/tjumx/OneDrive - University of Missouri/data/Course/2302-Computer-Vision-CS-ECE-8690/CV2023_HW4B/input'
OUTPUT_PATH = 'c:/Users/tjumx/OneDrive - University of Missouri/data/Course/2302-Computer-Vision-CS-ECE-8690/CV2023_HW4B/output'


def main():
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    flist = [f for f in os.listdir(INPUT_PATH) if f.endswith('.png')]
    flist = sorted(flist)
    n = len(flist)

    for ALPHA in ALPHA_list:
        for TM in TM_list:
            # print the parameters
            print(f'ALPHA = {ALPHA}, TM = {TM}\n')
            # Read the first image and initialize the model
            im = cv2.imread(os.path.join(INPUT_PATH, flist[0]))
            bg_model = BGSubModel(im, ALPHA, TM)

            # Set up the VideoWriter objects
            # the name should contain the parameters of the model, ALPHA and TM
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            video_file = os.path.join(OUTPUT_PATH, f'op_{ALPHA}_{TM}.avi')
            video_writer = cv2.VideoWriter(
                video_file, fourcc, 10, (im.shape[1], im.shape[0]))

            bgmean_video_file = os.path.join(
                OUTPUT_PATH, f'op_bg_{ALPHA}_{TM}.avi')
            bgmean_video_writer = cv2.VideoWriter(
                bgmean_video_file, fourcc, 10, (im.shape[1], im.shape[0]))

            fgmask_video_file = os.path.join(
                OUTPUT_PATH, f'op_fg_{ALPHA}_{TM}.avi')
            fgmask_video_writer = cv2.VideoWriter(
                fgmask_video_file, fourcc, 10, (im.shape[1], im.shape[0]))

            # Main loop
            for fr in range(n):
                # Read the image
                im = cv2.imread(os.path.join(INPUT_PATH, flist[fr]))

                # Classify the foreground using the model & Update the model with the new image
                fg_mask = bg_model.classify(im)
                bg_model.update(im)

                # Display the input frame, background model, and foreground mask
                # add frame number, ALPHA, and TM to the top left of the image
                im_draw = im.copy()
                cv2.putText(im_draw, f'Frame {fr+1}/{n}', (im.shape[1]//15, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(im_draw, f'ALPHA = {ALPHA}, TM = {TM}', (im.shape[1]//15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                bg_draw = bg_model.mean.astype('uint8')
                cv2.putText(bg_draw, f'Frame {fr+1}/{n}', (im.shape[1]//15, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(bg_draw, f'ALPHA = {ALPHA}, TM = {TM}', (im.shape[1]//15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                fg_draw = fg_mask.copy()
                cv2.putText(fg_draw, f'Frame {fr+1}/{n}', (im.shape[1]//15, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(fg_draw, f'ALPHA = {ALPHA}, TM = {TM}', (im.shape[1]//15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Input Frame', im_draw)
                cv2.imshow('Background Model', bg_draw)
                cv2.imshow('Foreground Mask', fg_draw)

                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Save the results for specific frames
                if fr in [5, 100, 400]:
                    fname = f'FGmask_{ALPHA}_{TM}_{flist[fr]}'
                    fname_wpath = os.path.join(OUTPUT_PATH, fname)
                    cv2.imwrite(fname_wpath, fg_draw)

                    fname = f'BGmean_{ALPHA}_{TM}_{flist[fr]}'
                    fname_wpath = os.path.join(OUTPUT_PATH, fname)
                    cv2.imwrite(fname_wpath, bg_draw)

                # Write the current frame and FGmask to the videos
                video_writer.write(im_draw)
                bgmean_video_writer.write(bg_draw)
                fgmask_video_writer.write(fg_draw)

                # Print the progress using the same line
                print(f'Frame {fr+1}/{n}', end='\r')

            video_writer.release()
            fgmask_video_writer.release()
            cv2.destroyAllWindows()
            # print the status
            print('Done                    \n')


if __name__ == '__main__':
    main()


# %%
