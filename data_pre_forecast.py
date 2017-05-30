import cv2
import numpy as np
import os
import datetime
import time
import pandas as pd
import utility


def main():
    # This is for when we haven't got all the images
    # Find all image filename
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # image_folder = os.path.join(dir_path , "fisheye_images")
    # image_names = os.listdir(image_folder)
    # image_names_format = "%Y%m%d_%H%M%S.jpg"
    #
    # number_images = len(image_names)
    # output_img_shape = [60, 60, 3]
    # all_images = np.ndarray([number_images] + output_img_shape,dtype='uint8')
    # pv_outputs = np.zeros(number_images)
    # all_times = np.zeros(number_images).astype(datetime.datetime)
    # mono_pv_csv = pd.read_csv("mono_pv_output.csv")
    #
    # curr_time = time.process_time()
    # for i in range(number_images):
    #     # Read in image file and the associated time stamp
    #     img = cv2.imread(os.path.join(image_folder, image_names[i]))
    #     img = utility.crop_central_image(img,img.shape[0],img.shape[0]) # crop only the central square
    #
    #     # resizing the image to output_img_shape
    #     resizing_ratio = output_img_shape[0] / img.shape[0]
    #     all_images[i] = cv2.resize(img, None, fx=resizing_ratio, fy=resizing_ratio)
    #     all_times[i] = datetime.datetime.strptime(image_names[i], image_names_format)
    #     pv_outputs[i] = find_pv_output(all_times[i], mono_pv_csv)
    #
    #     if i%10 ==0:
    #         print('image processed: ',i, '/',number_images)
    #         print('pv value:',pv_outputs[i])
    #
    # print(np.sum(np.isnan(pv_outputs)),' images do not have matching pv output')
    # valid_indices = np.logical_not(np.isnan(pv_outputs))
    # all_images = all_images[valid_indices, :, :, :]
    # all_times = all_times[valid_indices]
    # pv_outputs = pv_outputs[valid_indices]
    #
    # np.save('timestamps.npy',all_times)
    # np.save('images.npy', all_images)
    # np.save('pv_outputs', pv_outputs)
    #
    # print(time.process_time()-curr_time)

    tic = time.process_time()
    input_amount = 5
    input_interval = datetime.timedelta(minutes=1)
    output_horizon = datetime.timedelta(minutes=5)

    # load data
    all_times = np.load('timestamps.npy')
    all_images = np.load('images.npy')
    pv_persistent = np.load('pv_outputs.npy')
    mono_pv_csv = pd.read_csv("mono_pv_output.csv")

    # Initialization
    num_imgs,H,W,C = all_images.shape
    img_stack = np.zeros([H,W,C*input_amount],dtype='uint8')
    img_stacks = np.zeros([num_imgs,H,W,C*input_amount],dtype='uint8')
    pv_future= np.zeros([num_imgs])
    time_shift = np.arange(0, input_amount) * input_interval

    for i in range(num_imgs):
        last_time_cand = all_times[i]
        imgs_found = True
        for j in range(input_amount):
            curr_time = last_time_cand - time_shift[j]
            # is there image at curr_time?
            if curr_time in all_times:
                img_stack[:,:,j*C:(j+1)*C] = all_images[curr_time==all_times]
            else:
                imgs_found = False
                break
        if imgs_found:
            future_time = last_time_cand + output_horizon
            pv_future[i] = find_pv_output(future_time, mono_pv_csv)
        else:
            pv_future[i] = None
        img_stacks[i] = img_stack
        print('time candidate processed: ', i, '/', num_imgs)

    valid_indices = np.logical_not(np.isnan(pv_future))
    print('total valid sample:',np.sum(valid_indices))
    img_stacks = img_stacks[valid_indices, :, :, :]
    all_times = all_times[valid_indices]
    pv_future = pv_future[valid_indices]
    pv_persistent = pv_persistent[valid_indices]

    np.save('fore_times.npy',all_times)
    np.save('image_stacks.npy', img_stacks)
    np.save('pv_future.npy', pv_future)
    np.save('pv_persistent.npy',pv_persistent)
    print('computation time: ',time.process_time()-tic)


def find_pv_output(dt, pv_data):
    line = pv_data[pv_data["unixTS"] == dt.timestamp()]
    try:
        return line["Value"].iloc[0]
    except:
        return None


if __name__ == '__main__':
    main()
