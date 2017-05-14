import cv2
import datetime
import os
import numpy as np


def snap_from_video(video_path, capture_interval, image_folder, time_format="%Y%m%d_%H%M%S"):
    video_filename = os.path.basename(video_path)
    video_filename_format = "%Y%m%d-%H%M%S-01.avi"

    # snap images from the video file
    cap = cv2.VideoCapture(video_filename)
    initial_time = datetime.datetime.strptime(video_filename, video_filename_format)

    video_pos_time = 0  # in ms
    curr_time = initial_time

    while cap.isOpened():
        cap.set(0, video_pos_time)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(image_folder + curr_time.strftime(time_format) + ".jpg", frame)
            video_pos_time += capture_interval.total_seconds() * 1000  # in ms
            curr_time += capture_interval
        else:
            break
    end_time = curr_time - capture_interval

    return initial_time, end_time


def snap_seg_from_video(video_path, capture_interval, capture_length, image_folder, time_format="%Y%m%d_%H%M%S"):
    video_filename = os.path.basename(video_path)
    video_filename = video_filename.split('_')[3]  # this is only the case under current naming protocol
    video_filename_format = "%Y%m%d%H%M%S"

    # snap images from the video file
    cap = cv2.VideoCapture(video_path)
    initial_time = datetime.datetime.strptime(video_filename, video_filename_format)

    video_pos_time = 0  # in ms
    curr_time = initial_time

    while curr_time <= initial_time + capture_length + 2 * capture_interval:  # buffer zone
        cap.set(0, video_pos_time)
        ret, frame = cap.read()
        if not ret:
            print('capture length is longer than video length!')
            break
        else:
            cv2.imwrite(image_folder + curr_time.strftime(time_format) + ".jpg", frame)
            video_pos_time += capture_interval.total_seconds() * 1000  # in ms
            curr_time += capture_interval

    end_time = curr_time - capture_interval
    return initial_time, end_time


def documentation(model_name, rmse, prev_image_time, next_image_time, output_interval, horizon, elapsed_time,
                  resizing_ratio=1, time_format="%Y%m%d_%H%M%S"):
    doc = open("documentation.txt", "a")
    doc.write("Model name: " + model_name + "\n")
    doc.write("Simulation completion time: " + datetime.datetime.now().strftime(time_format) + "\n")
    doc.write("Optical flow image 1: " + prev_image_time.strftime(time_format) + "\n")
    doc.write("Optical flow image 2: " + next_image_time.strftime(time_format) + "\n")
    doc.write("Output interval: " + str(output_interval.total_seconds()) + " s\n")
    doc.write("Forecast horizon: " + str(horizon.total_seconds()) + " s\n")
    doc.write("Resizing ratio: " + str(resizing_ratio) + "\n")
    doc.write("Computation time: " + str(elapsed_time) + "\n")
    doc.write("RMSE: \n")
    rmse.tofile(doc, " ")
    doc.writelines(["\n", "\n"])
    doc.close()


def documation_for_models(method_names, initial_time, rmse_models_days, project_path, time_format="%Y%m%d_%H%M%S"):
    # Handle the case when rmse_persistence is passed in, a two dimensional array
    if rmse_models_days.ndim == 2:
        rmse_models_days = np.expand_dims(rmse_models_days, axis=2)
        method_names = [method_names, 'place_holder']

    for i in range(0, rmse_models_days.shape[2]):  # cycle through the models
        simulation_name = method_names[i] + '' + datetime.datetime.now().strftime(time_format)
        np.save(project_path + simulation_name, rmse_models_days[:, :, i])
        doc = open(project_path + simulation_name + '.txt', 'w')
        doc.write('Simulation name: ' + simulation_name + "\n")
        doc.write('Benchmark segments used (totaling ' + str(len(initial_time)) + '): \n')

        # write down all the video segments used in testing of the model
        for j in range(0, len(initial_time)):
            doc.write(initial_time[j].strftime(time_format) + '\n')

        doc.write("RMSE: \n")
        rmse_models_days[:, :, i].tofile(doc, " ")
        doc.close()  # finish documentation of one model


def crop_central_image(img, x_length, y_length):
    image_shape = np.array(img.shape[0:1])
    cropped_shape = np.array([x_length, y_length])
    x_low, y_low = ((image_shape - cropped_shape) / 2).astype('int')
    x_high, y_high = ((image_shape + cropped_shape) / 2).astype('int')
    img = img[x_low:x_high, y_low:y_high, :]
    return img


def read_flo_file(file_path):
    flo_file = open(file_path, 'rb')
    tag = np.fromfile(flo_file, 'float32', 1)
    width = np.fromfile(flo_file, 'int32', 1)
    width = width[0]
    height = np.fromfile(flo_file, 'int32', 1)
    height = height[0]
    # print(tag, width, height)

    flow = np.fromfile(flo_file, 'float32', height * width * 2)
    flow = np.reshape(flow, (height, width, 2))
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    flo_file.close()
    return u, v

