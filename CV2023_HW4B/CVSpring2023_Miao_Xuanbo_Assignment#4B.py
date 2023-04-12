# %% Import Packages
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% Background Subtraction Model


class BGSubModel:

    # The model is initialized with the first frame.
    def __init__(self, first_frame, alpha, tm, true_bg):
        self.mean = np.float32(first_frame)
        self.var = np.ones_like(self.mean) * 128
        self.alpha = alpha
        self.tm = tm
        self.true_bg = np.float32(true_bg)

    # Classify the current frame as foreground or background
    def classify(self, current_frame):
        diff = np.abs(np.float32(current_frame) - self.mean)
        self.fg_mask = np.where(
            diff > (self.tm * np.sqrt(self.var)), 255, 0).astype(np.uint8)

        diff_true = np.abs(np.float32(current_frame) - self.true_bg)
        self.true_fg_mask = np.where(
            diff_true > (self.tm * np.sqrt(self.var)), 255, 0).astype(np.uint8)

        return self.fg_mask

    def evaluate(self):
        mse_bg = np.mean((self.true_bg - self.mean) ** 2)
        max_val_bg = np.max((self.true_bg.max(), self.mean.max()))
        score_bg = 1 - np.sqrt((mse_bg / (max_val_bg ** 2)))

        mse_fg = np.mean((self.true_fg_mask - self.fg_mask) ** 2)
        max_val_fg = np.max((self.true_fg_mask.max(), self.fg_mask.max()))
        # considering fg is only a very small part of the image, so we add a scale factor to make the score more reasonable
        factor = 100
        score_fg = 1 - np.sqrt((mse_fg / (max_val_fg ** 2)))*factor

        return score_bg, score_fg

    # Update the model with the current frame
    def update(self, current_frame):
        inv_alpha = 1 - self.alpha
        self.mean = inv_alpha * self.mean + \
            self.alpha * np.float32(current_frame)
        self.var = inv_alpha * self.var + \
            self.alpha * (np.float32(current_frame) - self.mean) ** 2


def combine_images(output_path, alpha, tm, frame_numbers, file_prefixes):
    images = []

    for fr in frame_numbers:
        row_images = []
        for prefix in file_prefixes:
            fname = f'{prefix}_{alpha}_{tm}_{fr}.png'
            fname_wpath = output_path+'/'+fname
            img = cv2.imread(fname_wpath)
            row_images.append(img)
            os.remove(fname_wpath)

        row = np.hstack(row_images)
        images.append(row)

    final_image = np.vstack(images)
    return final_image


# %% Main
# Parameters
ALPHA_list = [0.01, 0.003, 0.001, 0.0003, 0.0001]
TM_list = [2, 3, 4]

# Files & Folders
INPUT_PATH = './CV2023_HW4B/input'
OUTPUT_PATH = './CV2023_HW4B/output'
BG_true_PATH = './CV2023_HW4B/input/GT_CAVIAR1.png'


def main():
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    flist = [f for f in os.listdir(INPUT_PATH) if f.endswith('.png')]
    flist = sorted(flist)
    n = len(flist)

    true_bg = cv2.imread(BG_true_PATH)

    for ALPHA in ALPHA_list:
        for TM in TM_list:
            # print the parameters
            print(f'ALPHA = {ALPHA}, TM = {TM}')
            # Read the first image and initialize the model
            im = cv2.imread(os.path.join(INPUT_PATH, flist[0]))
            bg_model = BGSubModel(im, ALPHA, TM, true_bg)

            # Set up the VideoWriter objects
            # the name should contain the parameters of the model, ALPHA and TM
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            video_file = os.path.join(
                OUTPUT_PATH, f'{ALPHA}_{TM}.avi')
            video_writer = cv2.VideoWriter(
                video_file, fourcc, 10, (im.shape[1]*3, im.shape[0]))

            bg_scores = []
            fg_scores = []
            # Main loop
            for fr in range(n):
                # Read the image
                im = cv2.imread(os.path.join(INPUT_PATH, flist[fr]))

                # Classify the foreground using the model & Update the model with the new image
                fg_mask = bg_model.classify(im)
                bg_model.update(im)

                # Display the input frame, background model, and foreground mask
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

                fg_draw = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
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
                    fname = f'FGmask_{ALPHA}_{TM}_{fr}.png'
                    fname_wpath = os.path.join(OUTPUT_PATH, fname)
                    cv2.imwrite(fname_wpath, fg_draw)

                    fname = f'BGmean_{ALPHA}_{TM}_{fr}.png'
                    fname_wpath = os.path.join(OUTPUT_PATH, fname)
                    cv2.imwrite(fname_wpath, bg_draw)

                # Write the current frame and FGmask to the videos
                frame = np.hstack(
                    (im_draw, bg_draw, cv2.cvtColor(fg_draw, cv2.COLOR_GRAY2BGR)))
                video_writer.write(frame)

                # Print the progress using the same line
                score_bg, score_fg = bg_model.evaluate()
                bg_scores.append(np.mean(score_bg))
                fg_scores.append(np.mean(score_fg))

                print(
                    f'Frame: {fr}/{n}, bg%: {np.mean(score_bg):.5f}, fg%: {np.mean(score_fg):.5f}', end='\r')

            # conbine these eix images into one image
            frame_numbers = [5, 100, 400]
            file_prefixes = ['FGmask', 'BGmean']
            image = combine_images(
                OUTPUT_PATH, ALPHA, TM, frame_numbers, file_prefixes)

            fname = f'{ALPHA}_{TM}.jpg'
            fname_wpath = os.path.join(OUTPUT_PATH, fname)
            cv2.imwrite(fname_wpath, image)

            cv2.destroyAllWindows()
            video_writer.release()
            cv2.destroyAllWindows()

            # print the status
            print(
                f'Done, bg%: {np.mean(score_bg):.5f}, fg%: {np.mean(score_fg):.5f}   \n')

            plt.plot(range(0, fr + 1), bg_scores, label='Background Score')
            plt.plot(range(0, fr + 1), fg_scores, label='Foreground Score')
            plt.xlabel('Frame Number')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(os.path.join(
                OUTPUT_PATH, f'op_curve_{ALPHA}_{TM}.png'))
            # then end this plt
            plt.close()


if __name__ == '__main__':
    main()
