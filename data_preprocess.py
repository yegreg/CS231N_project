import cv2
import numpy as np
import os
import datetime
import time
import pandas as pd
import utility


def main():
    # Find all image filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_folder = os.path.join(dir_path , "fisheye_images")
    image_names = os.listdir(image_folder)
    image_names_format = "%Y%m%d_%H%M%S.jpg"

    number_images = len(image_names)
    output_img_shape = [60, 60, 3]
    all_images = np.ndarray([number_images] + output_img_shape,dtype='uint8')
    pv_outputs = np.zeros(number_images)
    mono_pv_csv = pd.read_csv("mono_pv_output.csv")

    curr_time = time.process_time()
    for i in range(number_images):
        # Read in image file and the associated time stamp
        img = cv2.imread(os.path.join(image_folder, image_names[i]))
        img = utility.crop_central_image(img,img.shape[0],img.shape[0]) # crop only the central square

        # resizing the image to output_img_shape
        resizing_ratio = output_img_shape[0] / img.shape[0]
        all_images[i] = cv2.resize(img, None, fx=resizing_ratio, fy=resizing_ratio)
        image_time = datetime.datetime.strptime(image_names[i], image_names_format)
        pv_outputs[i] = find_pv_output(image_time, mono_pv_csv)

        if i%10 ==0:
            print('image processed: ',i, '/',number_images)
            print('pv value:',pv_outputs[i])

    print(np.sum(np.isnan(pv_outputs)),' images do not have matching pv output')
    valid_indices = np.logical_not(np.isnan(pv_outputs))
    all_images = all_images[valid_indices, :, :, :]
    pv_outputs = pv_outputs[valid_indices]

    np.save('images.npy', all_images)
    np.save('pv_outputs', pv_outputs)

    print(time.process_time()-curr_time)


def find_pv_output(dt, pv_data):
    line = pv_data[pv_data["unixTS"] == dt.timestamp()]
    try:
        return line["Value"].iloc[0]
    except:
        return None


if __name__ == '__main__':
    main()
