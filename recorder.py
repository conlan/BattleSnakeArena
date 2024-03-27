from PIL import Image
from PIL import Image, ImageDraw, ImageFont

from constants import LocalDirection

import os
import imageio
import numpy as np

class Recorder():
    def output_to_video(self, observations, output_file):
        if (len(observations) == 0):
            print("No observations to output")
            return

        with imageio.get_writer(output_file, fps=8) as writer:  # Adjust the fps as needed
            np_img_array = self._convert_images_to_np_array(observations)            

            for np_image in np_img_array:
                writer.append_data(np_image)

        print("Video saved to: " + output_file)

    def output_to_frames(self, observations, output_dir):
        if (len(observations) == 0):
            print("No observations to output")
            return
        
        # delete the folder if it exists
        if os.path.exists(output_dir):
            print(f"Deleting existing frames in {output_dir}")
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
        
        # make the output dir if it doesn't exist        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)                            

        # convert the observations to np array
        np_img_array = self._convert_images_to_np_array(observations)

        # output each np array frame to a file
        for i in range(len(np_img_array)):
            img = Image.fromarray(np_img_array[i])

            # pad i to 5 digits
            frame_index = str(i).zfill(5)
            
            img.save(f'{output_dir}/frame_{frame_index}.png')

        print(f"Frames saved to: {output_dir}")

    def _convert_images_to_np_array(self, observations):
        SINGLE_FRAME_SIZE = 67

        FRAME_PADDING = 50
        
        FINAL_FRAME_SIZE = SINGLE_FRAME_SIZE + FRAME_PADDING * 2

        DEFAULT_FONT = ImageFont.truetype("assets/OpenSans-Bold.ttf", 10)

        # Define text and font settings    
        active_font_color = (255, 0, 0)  # Red
        inactive_font_color = (0, 0, 0)  # Black

        np_array = []

        for frame_obj in observations:
            image = frame_obj['image']

            q_values = frame_obj['q_values'] if 'q_values' in frame_obj else None
            actor_probs = frame_obj['actor_probs'] if 'actor_probs' in frame_obj else None
            local_dir_of_move_made = frame_obj['action'] if 'action' in frame_obj else 0
            move_made = frame_obj['move'] if 'move' in frame_obj else None

            # if we don't have q_values (DDQN) but we have actor_probs (PPO) then use that instead
            if (q_values is None) and (actor_probs is not None):
                q_values = actor_probs
            
            # paste image onto bigger image
            final_image = Image.new("RGB", (image.width + FRAME_PADDING * 2, image.height + FRAME_PADDING * 2), (255, 255, 255))
            
            # paste image center of bigger image
            final_image.paste(image, (FRAME_PADDING, FRAME_PADDING))

            if q_values is not None:
                # print(q_values)

                draw = ImageDraw.Draw(final_image)

                for i in range(len(q_values)):
                    was_chosen_move = (i == local_dir_of_move_made)

                    text = "{:.3f}".format(q_values[i])
                    
                    x = 0
                    y = 0

                    write_top = False
                    write_left = False
                    write_right = False
                    write_down = False

                    # determine the coordinates to draw the text based on what direction this is
                    # and what direction the snake went
                    if (local_dir_of_move_made == LocalDirection.STRAIGHT):
                        if (move_made == 'up'):
                            if (i == LocalDirection.STRAIGHT):
                                write_top = True
                            elif (i == LocalDirection.RIGHT):
                                write_right = True
                            elif (i == LocalDirection.LEFT):
                                write_left = True
                        elif (move_made == 'left'):
                            if (i == LocalDirection.STRAIGHT):
                                write_left = True
                            elif (i == LocalDirection.RIGHT):
                                write_top = True
                            elif (i == LocalDirection.LEFT):
                                write_down = True
                        elif (move_made == 'right'):
                            if (i == LocalDirection.STRAIGHT):
                                write_right = True
                            elif (i == LocalDirection.RIGHT):
                                write_down = True
                            elif (i == LocalDirection.LEFT):
                                write_top = True
                        elif (move_made == 'down'):
                            if (i == LocalDirection.STRAIGHT):
                                write_down = True
                            elif (i == LocalDirection.RIGHT):
                                write_left = True
                            elif (i == LocalDirection.LEFT):
                                write_right = True
                    elif (local_dir_of_move_made == LocalDirection.LEFT):
                        if (move_made == 'up'):
                            if (i == LocalDirection.STRAIGHT):
                                write_right = True
                            elif (i == LocalDirection.RIGHT):
                                write_down = True
                            elif (i == LocalDirection.LEFT):
                                write_top = True
                        elif (move_made == 'right'):
                            if (i == LocalDirection.STRAIGHT):
                                write_down = True
                            elif (i == LocalDirection.RIGHT):
                                write_left = True
                            elif (i == LocalDirection.LEFT):
                                write_right = True
                        elif (move_made == 'left'):
                            if (i == LocalDirection.STRAIGHT):
                                write_top = True
                            elif (i == LocalDirection.RIGHT):
                                write_right = True
                            elif (i == LocalDirection.LEFT):
                                write_left = True
                        elif (move_made == 'down'):
                            if (i == LocalDirection.STRAIGHT):
                                write_left = True
                            elif (i == LocalDirection.RIGHT):
                                write_top = True
                            elif (i == LocalDirection.LEFT):
                                write_down = True
                    elif (local_dir_of_move_made == LocalDirection.RIGHT):
                        if (move_made == 'up'):
                            if (i == LocalDirection.STRAIGHT):
                                write_left = True
                            elif (i == LocalDirection.RIGHT):
                                write_top = True
                            elif (i == LocalDirection.LEFT):
                                write_down = True
                        elif (move_made == 'right'):
                            if (i == LocalDirection.STRAIGHT):
                                write_top = True
                            elif (i == LocalDirection.RIGHT):
                                write_right = True
                            elif (i == LocalDirection.LEFT):
                                write_left = True
                        elif (move_made == 'left'):
                            if (i == LocalDirection.STRAIGHT):
                                write_down = True
                            elif (i == LocalDirection.RIGHT):
                                write_left = True
                            elif (i == LocalDirection.LEFT):
                                write_right = True
                        elif (move_made == 'down'):
                            if (i == LocalDirection.STRAIGHT):
                                write_right = True
                            elif (i == LocalDirection.RIGHT):
                                write_down = True
                            elif (i == LocalDirection.LEFT):
                                write_top = True
                    
                    if (write_top):
                        x = FINAL_FRAME_SIZE // 2 - 10
                        y = FRAME_PADDING // 2
                    elif (write_left):
                        x = 5
                        y = FRAME_PADDING + SINGLE_FRAME_SIZE // 2
                    elif (write_right):
                        x = FRAME_PADDING + SINGLE_FRAME_SIZE + 10
                        y = FRAME_PADDING + SINGLE_FRAME_SIZE // 2
                    elif (write_down):
                        x = FINAL_FRAME_SIZE // 2 - 10
                        y = FINAL_FRAME_SIZE - FRAME_PADDING // 2
                    
                
                    draw.text((x, y), text, font=DEFAULT_FONT, fill=active_font_color if (was_chosen_move) else inactive_font_color)

            np_array.append(np.array(final_image))

        return np_array