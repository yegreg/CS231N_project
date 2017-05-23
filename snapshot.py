import utility
import os
import datetime


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    video_folder = dir_path + "\\input_videos\\"
    image_folder = os.path.join(dir_path, "fisheye_images")
    if not os.path.exists(image_folder): # make image folder if it didn't exist
        os.makedirs(image_folder)

    # define the parameter for snapshots from the videos
    capture_interval = datetime.timedelta(minutes=1)  # in s

    video_folder_months = os.listdir(video_folder)
    for i in range(4,len(video_folder_months)): # cycle through the months
        video_folder_date = os.listdir(os.path.join(video_folder, video_folder_months[i]))
        for j in range(0,len(video_folder_date)): # cycle through the dates
            curr_folder = os.path.join(video_folder,video_folder_months[i],video_folder_date[j])
            video_filenames = os.listdir(curr_folder)
            print('processing ',video_folder_months[i],'-',video_folder_date[j])
            for k in range(0, len(video_filenames)):
                # snap images from the video file
                video_path = os.path.join(curr_folder,video_filenames[k])
                initial_time_i, end_time = utility.snap_all_from_video_whole_minute(video_path, capture_interval, image_folder)
                print("processing image at time of",initial_time_i.strftime("%H%M%S"))


if __name__ == '__main__':
    main()
