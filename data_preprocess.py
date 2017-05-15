import cv2
import numpy as np
import os
import datetime
import pandas as pd


def main():
    # Find all image filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    project_name = "trial_on_dec_vid"
    project_path = dir_path + "\\" + project_name + "\\"
    image_folder = project_path + "fisheye_images\\"
    image_names = os.listdir(image_folder)
    image_names_format = "%Y%m%d_%H%M%S.jpg"

    number_images = len(image_names)
    output_img_shape = [60, 80, 3]
    all_images = np.zeros([number_images] + output_img_shape)
    pv_outputs = np.zeros(number_images)
    mono_pv = pd.read_csv("mono_pv_output.csv")
    valid_indices = []

    for i in range(number_images):
        # Read in image file and the associated time stamp
        img = cv2.imread(os.path.join(image_folder, image_names[i]))

        # resizing the image to output_img_shape
        resizing_ratio = output_img_shape[0] / img.shape[0]
        all_images[i] = cv2.resize(img, None, fx=resizing_ratio, fy=resizing_ratio)
        image_time = datetime.datetime.strptime(image_names[i], image_names_format)
        pv_output = find_pv_output(image_time, mono_pv)
        pv_outputs[i] = pv_output

        if pv_output is not None:
            valid_indices.append(i)
        else:
            print('no associated pv output for image: ',i)

        if i%10 ==0:
            print('image processed: ',i, '/',number_images)
            print('pv value:',pv_output)

    all_images = all_images[valid_indices, :, :, :]
    pv_outputs = pv_outputs[valid_indices]

    np.save('images.npy', all_images)
    np.save('pv_outputs', pv_outputs)


def find_pv_output(dt, pv_data):
    line = pv_data[pv_data["unixTS"] == dt.timestamp()]
    try:
        return line["Value"].iloc[0]
    except:
        return None


if __name__ == '__main__':
    main()
