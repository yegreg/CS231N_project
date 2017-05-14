import utility
import os
import datetime


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    project_name = "trial_on_dec_vid"
    project_path = dir_path + "\\" + project_name + "\\"
    video_folder = dir_path + "\\input_video\\"
    image_folder = project_path + "fisheye_images\\"
    video_folder_months = os.listdir(video_folder)

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # currently there is only one month in the video folder
    video_folder_date = os.listdir(os.path.join(video_folder, video_folder_months[0]))

    # define the parameter for snapshots from the videos
    time_format = "%Y%m%d_%H%M%S"
    capture_interval = datetime.timedelta(minutes=1)  # in s
    initial_time = []
    for i in range(0,len(video_folder_date)):
        curr_folder = os.path.join(video_folder,video_folder_months[0],video_folder_date[i])
        video_filename = os.listdir(curr_folder)
        for j in range(0, len(video_filename)):
            # snap images from the video file
            if video_filename[j] != "desktop.ini":
                video_path = os.path.join(curr_folder,video_filename[j])
                initial_time_i, end_time = utility.snap_all_from_video_whole_minute(video_path, capture_interval, image_folder)
                initial_time.append(initial_time_i)

if __name__ == '__main__':
    main()
