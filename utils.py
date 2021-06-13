import os

import imageio

import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def create_gif_from_images(dir_path):
    with imageio.get_writer('evolution.gif', mode='I') as writer:
        filenames = os.listdir(dir_path)
        # print(filenames[0].split("-"))
        # print(filenames)
        # filenames = filenames.sort(key=natural_keys)
        print(filenames)
        for filename in filenames:
            image = imageio.imread(dir_path + "/" + filename)
            writer.append_data(image)

create_gif_from_images("plots/temp")
