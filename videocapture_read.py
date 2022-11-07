
import sys
sys.path.insert(1, 'Lib')
sys.path.insert(1, 'Lib/ImageSimilarity')

import csv
import cv2
import os
#import main_multi
import shutil
import re
import time


FULL_DELETE = True #DO NOT SET THIS TO TRUE UNLESS THE USER WISHES TO WIPE DATA
#read settings
frame_interval = 10
#NOTE: a frame interval of 0 includes every frame
#      a frame interval of 10 will take one frame, then skip 9 frames,
#      processing a total of 10 frames
similarity_threshhold = 50 #in percentage

#video data 
video_file = 'sample.mp4'
pixel_width = 1600
pixel_height = 900
#set to 0 and 0 for no resize
dimensions = (pixel_width, pixel_height)

#directories
parent_directory = "processing/"
original_frame_directory = parent_directory + "original_frames/"
resized_frame_directory = parent_directory + "resized_frames/"
important_frame_directory = parent_directory + "important_frames/"
csv_directory = parent_directory + "csv/"
caption_directory = parent_directory + "captions/"

#prefixes
original_frame_prefix = "frame_"
resized_frame_prefix = "resized_frame_"
important_frame_prefix = "important_frame_"
caption_prefix = "caption_"

csv_base_filename = "csv_base_images.csv"
csv_comparison_filename = "csv_comparison_images.csv"

machine_directory = "machine_learning/"

def csv_stuff():
    # open the file in the write mode
    f = open('path/to/csv_file', 'w')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(row)

    # close the file
    f.close()


def create_basic_directories():
    """
    create basic directories to ensure smooth execution and organisation
    and when grabbing video frames
    """
    directories = [
      original_frame_directory,
      resized_frame_directory,
      important_frame_directory,
      csv_directory,
      caption_directory,
      ]
    try:
        os.mkdir(parent_directory)
    except:
        if FULL_DELETE:
            shutil.rmtree(parent_directory)
            os.mkdir(parent_directory)
        else:
            pass

    
    for i in directories:
        try:
            os.mkdir(i)
        except:
            shutil.rmtree(i)
            os.mkdir(i)

def get_frames(video_file, dimensions):
    """
    Inputs:
    - video_file: name of video file to be analysed
    - dimensions: dimensions of the video to be resized,
      if (0,0) is the input, do not resize
    Outputs:
    - frame_number: Number of frames
    - frame_name_format: filename format of files frames are written to
    """
    frame_name_format = resized_frame_prefix + "XXX" + ".png"


    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    captured_image = cv2.VideoCapture(video_file)
    fps = captured_image.get(cv2.CAP_PROP_FPS)
    # Check if camera opened successfully
    if not captured_image.isOpened(): 
      print("Error opening video stream or file")
      sys.exit()

    # Read until video is completed
    frame_number = 0
    while(captured_image.isOpened()):
      # Capture frame-by-frame
      read_success, frame = captured_image.read()
      if read_success:
        original_filename = original_frame_directory + original_frame_prefix + str(frame_number) + ".png"
        cv2.imwrite(original_filename, frame)
        resized_filename = resized_frame_directory + resized_frame_prefix + str(frame_number) + ".png"
        resized = cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)
        cv2.imwrite(resized_filename, resized)
##          for i in range(frame_interval):
##            read_success, frame = captured_image.read()
        frame_number += 1
      else: 
        break

    # Release the video capture object
    captured_image.release()

    return frame_number, frame_name_format, fps

def write_csv(number_of_frames, frame_name_format):
    """
    prepare csvs for image comparison
    """

    f_base = open(csv_directory + csv_base_filename, 'w', newline='')
    f_compare = open(csv_directory + csv_comparison_filename, 'w', newline='')
    writer_base = csv.writer(f_base)
    writer_compare = csv.writer(f_compare)

    writer_base.writerow(['id','url'])
    writer_compare.writerow(['id','url'])
    
    for i in range(number_of_frames-1):
        base_name = resized_frame_directory + frame_name_format.replace("XXX",str(i))
        compare_name = resized_frame_directory + frame_name_format.replace("XXX",str(i+1))

        writer_base.writerow([str(i), base_name])
        writer_compare.writerow([str(i), compare_name])
        
    f_base.close()
    f_compare.close()


def get_similarity(result):
    #print(result)
    calculated_similarity = "\"metrics\": \{\"psnr\": ([A-Za-z0-9\.]+)\}\}"
    similarity = re.findall(calculated_similarity, result)
    #print("sim:", similarity)
    a = similarity[0]
    if a == "Infinity":
        return 100
    else:
        return float(similarity[0])

def compare(frame_1, frame_2):
    
    command = "python \"Lib\\image-similarity-measures-master\\image_similarity_measures\\evaluate.py\" --org_img_path \"{}\" --pred_img_path \"{}\" --metric psnr".format(
        resized_frame_directory + frame_1, resized_frame_directory + frame_2)
    output = os.popen(command)
    print(command)
    a = output.read()
    return a

def get_important_frames(number_of_frames, frame_name_format):
    important_frames = [0]
    current_frame = 0
    for i in range(0, number_of_frames-frame_interval, frame_interval):
        result = compare(frame_name_format.replace("XXX", str(current_frame)), frame_name_format.replace("XXX", str(i+frame_interval)))
        #print("get sim results")
        print("compare: ", current_frame, i + frame_interval)
        sim = get_similarity(result)
        print(sim)
        #print(type(sim))
        #input()
        if sim < similarity_threshhold:
            important_frames.append(i+frame_interval)
            current_frame = i+frame_interval
    return important_frames

def caption(frame):
    command = "python image_caption.py caption " + resized_frame_directory + resized_frame_prefix + str(frame) + ".png"
    #print("command:", command)
    output = os.popen(command)
    data = output.read()
    #print("data:",data)
    caption_line = "###([A-Za-z0-9\.\s]+)###"
    caption = re.findall(caption_line, data)
    return caption

def add_caption(resized_frame_number, frame_name_format, caption):
    resized_frame = resized_frame_directory + frame_name_format.replace("XXX", str(resized_frame_number))
    captioned_frame = caption_directory + "captioned_frame_" + str(resized_frame_number) + ".png"
    im = cv2.imread(resized_frame)

    bordersize = 50
    border = cv2.copyMakeBorder(
        im,
        top=0,
        bottom=bordersize,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    image_capted = cv2.putText(border, "Image description: " + caption, (10,940), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(captioned_frame, image_capted)
    
def main():
    create_basic_directories()
    number_of_frames, frame_name_format, fps = get_frames(video_file, dimensions)
    #print(number_of_frames, frame_name_format)
    important_frames = get_important_frames(number_of_frames, frame_name_format) + [number_of_frames]
    #print(important_frames)
                
    for i in range(len(important_frames)-1):
        caption_to_add = caption(important_frames[i])[0]
        for j in range(important_frames[i], important_frames[i+1]):
            add_caption(j, frame_name_format, caption_to_add)
            
    new_frame_size = (1600, 950)
    
    fourcc = None
    if ".avi" in video_file:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    elif ".mp4" in video_file:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
    capt_file = "captioned_"
    vfi = video_file.rfind("\\")
    if vfi == -1:
        capt_file = "captioned_" + video_file[:]
    else:
        capt_file = video_file[:vfi] + "\\captioned_" + video_file[vfi+1:]
        
    VW = cv2.VideoWriter(capt_file, fourcc, fps, new_frame_size)
    for i in range(number_of_frames):
        captioned_frame = caption_directory + "captioned_frame_" + str(i) + ".png"
        try:
            VW.write(cv2.imread(captioned_frame))
        except:
            pass
    VW.release()

def testing():
    number_of_frames, frame_name_format, fps = 5804, "resized_frame_XXX.png", 30
    important_frames = [0, 3200, 3400, 3500, 3600, 3700, 4700, 5200, 5400, 5500, 5600, 5700, 5800] + [number_of_frames]
    
    ##create_basic_directories()
    ##number_of_frames, frame_name_format = get_frames(video_file, dimensions)
    ##important_frames = get_important_frames(number_of_frames, frame_name_format)
    #print(important_frames)

    for i in range(len(important_frames)-1):
        caption_to_add = caption(important_frames[i])[0]
        for j in range(important_frames[i], important_frames[i+1]):
            add_caption(j, frame_name_format, caption_to_add)
            if j == 300:
                break
        break
            
    new_frame_size = (1600, 950)
    
    fourcc = None
    if ".avi" in video_file:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    elif ".mp4" in video_file:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

    capt_file = "captioned_"
    vfi = video_file.rfind("\\")
    if vfi == -1:
        capt_file = "captioned_" + video_file[:]
    else:
        capt_file = video_file[:vfi] + "\\captioned_" + video_file[vfi+1:]
        
    capt_file = "captioned_"
    vfi = video_file.rfind("\\")
    if vfi == -1:
        capt_file = "captioned_" + video_file[:]
    else:
        capt_file = video_file[:vfi] + "\\captioned_" + video_file[vfi+1:]
        
    VW = cv2.VideoWriter(capt_file, fourcc, fps, new_frame_size)
    for i in range(number_of_frames):
        captioned_frame = caption_directory + "captioned_frame_" + str(i) + ".png"
        try:
            VW.write(cv2.imread(captioned_frame))
        except:
            pass
    VW.release()
        
#python Monika/image_caption.py train_custom 5 Flicker8k_Dataset_small Flickr_8k.trainImages_small.txt Flickr8k.token_small.txt Flickr_8k.testImages_small.txt
#main()

def get_args_from_cmd_line():
    global video_file
    if len(sys.argv) != 2:
        print("Invalid number of args")
        exit(1)
    video_file = sys.argv[1]
    try:
        with open(video_file, "rb") as f:
            pass
    except:
        print("Invalid file")
        exit(1)
        
    main()

get_args_from_cmd_line()
