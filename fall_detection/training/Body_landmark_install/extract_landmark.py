import os
import subprocess

# # change this to the path to your directory
# os.chdir('/home/tungtt/Desktop/Fall_Detection_Deep_Learning_Model/openpose')
root_dir = '/home/tungtt/Desktop/Fall_Detection_Deep_Learning_Model/data/lei2/Office/Office'

# video_file = 'video (10).mp4'
# frame_dir = '/home/tungtt/Desktop/Fall_Detection_Deep_Learning_Model/data/ur_dataset'
# json_dir = frame_dir
# output_video = 'adl_result_video (10).mp4'
# openpose_bin = './openpose.bin'
# model_folder = '/home/tungtt/Desktop/Fall_Detection_Deep_Learning_Model/openpose/models'
# model_path = './final_cnn_model.h5'

# openpose_cmd = f'./openpose.bin --video {frame_dir} --write_json {json_dir} --display 0 --render_pose 0 --model_folder /home/tungtt/Desktop/Fall_Detection_Deep_Learning_Model/openpose/models'

def extract_landmark():
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # get file name without extension
            file_name = os.path.splitext(file)[0]
            extension = os.path.splitext(file)[1].lower()

            if (extension == '.mp4' or extension == '.avi' or extension == '.mov') and (not os.path.isdir(os.path.join(subdir, file_name))):
                # create a folder with the same file name
                folder_dir = os.path.join(subdir, file_name)
                os.mkdir(folder_dir)

                # run the command to extract features
                cmd_str = './openpose.bin --video "' + os.path.join(subdir, file) + '" --write_json "' + folder_dir + '" --display 0 --render_pose 0 --net_resolution "-1x256" --model_folder /home/tungtt/Desktop/Fall_Detection_Deep_Learning_Model/openpose/models' 
                os.system(cmd_str)
                # subprocess.call(openpose_cmd, shell=True)
                

if __name__ == "__main__":
    extract_landmark()


