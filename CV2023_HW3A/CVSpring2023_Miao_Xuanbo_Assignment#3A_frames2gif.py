from PIL import Image
import os
import numpy as np

frames_dir = './frames/'
output_gif_path = 'output.gif'

frame_duration = 1000/60

frames = []

def main():
    for i in range(len(os.listdir(frames_dir))):
        
        frame_path = os.path.join(frames_dir, f'frame_{i}.png')
        frame = Image.open(frame_path)

        frame = frame.resize((640, 480))

        frames.append(frame)
        print(f"./frames/frame_{i}.png")

    frames[0].save(output_gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=frame_duration, loop=0)
