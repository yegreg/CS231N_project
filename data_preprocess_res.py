import cv2
import numpy as np
import os
import datetime
import pandas as pd
import utility
import time


def main():
    # Find all image filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_folder = os.path.join(dir_path, "fisheye_images")
    npy_folder = os.path.join(dir_path, "input_resolution")
    image_names = os.listdir(image_folder)
    image_names_format = "%Y%m%d_%H%M%S.jpg"
    number_images = len(image_names) - 1

    # generate an array of output image shape
    min_len = 32
    max_len = 128
    step_len = 8
    output_img_len = np.arange(min_len, max_len + 1, step_len)
    pv_outputs = np.zeros(number_images, dtype='float32')
    all_times = np.zeros(number_images).astype(datetime.datetime)
    mono_pv_csv = pd.read_csv("mono_pv_output.csv")

    tic = time.process_time()
    # initialize a dict of np arrays to store resized images
    img_resized_dict = {}
    for i in range(output_img_len.size):
        output_img_shape = [output_img_len[i], output_img_len[i], 3]
        img_resized_dict[output_img_len[i]] = np.zeros([number_images] + output_img_shape, dtype='uint8')

    # resize each image to different sizes and store them in the array
    for i in range(number_images):
        # Read in image file and the associated time stamp
        img = cv2.imread(os.path.join(image_folder, image_names[i]))
        img = utility.crop_central_image(img, img.shape[0], img.shape[0])  # crop only the central square
        all_times[i] = datetime.datetime.strptime(image_names[i], image_names_format)
        pv_outputs[i] = find_pv_output(all_times[i], mono_pv_csv)

        # if current output is none don't bother resizing images
        if np.isnan(pv_outputs[i]):
            continue

        # if current output is not nan then do the resizing
        for j in range(output_img_len.size):
            # resizing the image to output_img_shape
            resizing_ratio = output_img_len[j] / img.shape[0]
            img_resized_dict[output_img_len[j]][i] = cv2.resize(img, None, fx=resizing_ratio, fy=resizing_ratio)

        if i % 10 == 0:
            print('image processed: ', i, '/', number_images)
            print('pv value:', pv_outputs[i])

    print(np.sum(np.isnan(pv_outputs)), ' images do not have matching pv output')
    valid_indices = np.logical_not(np.isnan(pv_outputs))
    all_times = all_times[valid_indices]
    pv_outputs = pv_outputs[valid_indices]

    np.save(npy_folder + '\\timestamps.npy', all_times)
    np.save(npy_folder + '\\pv_outputs', pv_outputs)

    # save valid images
    for i in range(output_img_len.size):
        img_resized_valid = img_resized_dict[output_img_len[i]][valid_indices, :, :, :]
        np.save(npy_folder + '\\images' + str(output_img_len[i]) + '.npy', img_resized_valid)

    print('process time: ', time.process_time() - tic)


def find_pv_output(dt, pv_data):
    line = pv_data[pv_data["unixTS"] == dt.timestamp()]
    try:
        return line["Value"].iloc[0]
    except:
        return None


if __name__ == '__main__':
    main()
