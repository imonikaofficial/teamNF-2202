import sys
import os
import json
import moviepy.editor as mp

import blobupload as bu
import transcription_services as ts

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


# ---------------- initialize variables ----------------
speech_key, service_region = os.getenv('SPEECH_KEY'),"eastasia"
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
target_container = 'https://02projectstorage.blob.core.windows.net/' # to be used by appending target container to string
send_request_transcription = "https://" + service_region + ".api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions"
# 'mp4', 'm4a', 'm4v', 'f4v', 'f4a', 'm4b', 'm4r', 'f4b', 'mov', 'wmv', 'wma', 'asf*', 'ogg', 'oga', 'ogv', 'ogx', '3gp', '3gp2', '3g2', '3gpp', '3gpp2'
accepted_audio = ['m4a']
accepted_video  = ['mkv', 'mp4']

# ---------------- function menus ----------------
# ------ main menu in general ------
def main_main_menu():
    while True:
        print("----2202 AI Video Analysis----")
        print("--Main Menu--")
        print("0. Exit Program\n1. Transcribe Files\n2. AI Video Recognition")
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > 2 or user_in < 0:
                raise Exception()
            else:
                break
        except:
            print("Enter a valid option!")
    if user_in == 0:
        sys.exit()
    elif user_in == 1:
        main_menu()
    elif user_in == 2:
        video_menu()
    
# ------ main menu for AI Video ------ 
def video_menu():
    while True:
        print("--Video Recognition Menu--")
        print("0. Back To Main Menu\n1. Select File/Folder for Processing")
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > 3 or user_in < 0:
                raise Exception()
            else:
                break
        except:
            print("Enter a valid option!")
    if user_in == 0:
        main_main_menu()
    elif user_in == 1:
        select_video()

# ------ menu to select video file/folder ------
def select_video():
    while True:
        print("Only single files accepted for now.")
        print("Type 'x' to cancel")
        user_in = input("Enter File Path: ")
        if user_in == 'x':
            video_menu()
        else: 
            if os.path.isfile(user_in):
                os.system("python ./videocapture_read.py " + '"' + user_in + '"')
                input("Video Processing Done. Press Enter to Continue...")
                break
            else:
                print("Invalid file path")
    video_menu()

# ------ main menu for transcription ------ 
def main_menu():
    while True:
        print("--Transcription Menu--")
        print("0. Back To Main Menu\n1. Azure Storage Services\n2. Transcribe Files in Azure Storage\n3. Convert Video/Audio to WAV format for transcription")
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > 3 or user_in < 0:
                raise Exception()
            else:
                break
        except:
            print("Enter a valid option!")
    if user_in == 0:
        main_main_menu()
    elif user_in == 1:
        storage_menu()
    elif user_in == 2:
        transcription_menu()
    elif user_in == 3:
        convert_menu()

# ------ convert files menu ------ 
def convert_menu():
    try:
        if os.path.isdir('./converted_audio'):
            pass
        else:
            os.mkdir('./converted_audio')
    except:
        print("Unable to access ./converted_audio directory")
        sys.exit()
    while True:
        print("--Convert Video/Audio menu--")
        print("All files converted will be stored in ./converted_audio")
        print("\tAccepted Video Formats:", *accepted_video, sep=' ')
        print("\tAccepted Audio Formats:", *accepted_audio, sep=' ')
        print("0. Back to Main Menu\n1. Specify folder/file for conversion\n2. View Converted files")
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > 2 or user_in < 0:
                raise Exception()
            else:
                break
        except:
            print("Enter a valid option!")
            input("Press Enter to Continue...")
    if user_in == 0:
        main_menu()
    elif user_in == 1:
        conversion_menu()
    elif user_in == 2:
        convert_view_menu()

# ------ convert video/audio to wav ------
def conversion_menu():
    print("If path is file, convert file to audio.\nIf path is folder, convert all files in folder to audio\nType 'x' to cancel")
    filepath = input("Enter file path: ")
    if filepath == "x":
        pass
    elif os.path.isfile(filepath):
        helper_conversion(filepath)
        input("Press Enter to Continue...")
    elif os.path.isdir(filepath):
        folder_contents = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
        for f in folder_contents:
           helper_conversion(os.path.join(filepath, f))
           input("Press Enter to Continue...")
    else:
        print("Not a file nor folder, try again.")
        input("Press Enter to Continue...")
    convert_menu()

def helper_conversion(filepath): #helper function for conversion
    if filepath.split('.')[-1] in accepted_audio:
        ts.m4a_to_wav(filepath)
    elif filepath.split('.')[-1] in accepted_video:
        ts.video_to_wav(filepath)
    else:
        print("File " + filepath + " not an accepted audio/video format.")

# ------ convert view menu ------
def convert_view_menu():
    filepath = './converted_audio/'
    folder_contents = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
    for f in folder_contents:
        print("\t" + f)
    input("Press Enter to Continue...")
    convert_menu()

# ------ transcription menu ------ 
def transcription_menu():
    while True:
        print("--Transciption Menu--")
        print("0. Back to Main Menu\n1. Transcribe Container\n2. View Current Transciption Requests\n3. Delete Transciption Request\n4. Refresh Current Transcription Request Status")
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > 4 or user_in < 0:
                raise Exception()
            else:
                break
        except:
            print("Enter a valid option!")
            input("Press Enter to Continue...")
    if user_in == 0:
        main_menu()
    elif user_in == 1:
        transcription_container_menu()
    elif user_in == 2:
        transcription_request_menu()
    elif user_in == 3:
        transcription_delete_menu()
    elif user_in == 4:
        refresh_transcriptions()

# ------ transciption container menu ------ 
def transcription_container_menu():
    while True:
        print("Select Container for Transciption: ")
        print("0. Back to Transciption Menu")
        name_list = bu.list_containers(blob_service_client)
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > len(name_list) or user_in < 0:
                raise Exception()
            else:
                break
        except:
            print("Enter a valid option!")
            input("Press Enter to Continue...")
    if user_in == 0:
        transcription_menu()
    elif user_in > 0 and user_in <= len(name_list):
        container_name = name_list[user_in - 1]
        response = ts.create_batch_transription(target_container + container_name, send_request_transcription)
        #print(response.json())
        #print(response.status_code())
        initial_dict = ts.response_initial(response)
        print("Transcription ID: " + initial_dict.get('transcription_id'))
        print("Status: " + initial_dict.get('status'))
        print("Summary of Results link: " + initial_dict.get('files'))
        input("Press Enter to Continue...")
        transcription_menu()
    else:
        print("Error with user input")
        input("Press Enter to Continue...")

# ------ transcription request menu ------ 
def transcription_request_menu():
    while True:
        print("Select Transciption for Viewing: ")
        print("0. Back to Transciption Menu")
        file_line = []
        count = 1
        with open('./transcriptions.txt', "r") as outfile:
            file_line = outfile.readlines()
        for i in file_line:
            str_json = json.loads(i)
            print(str(count) + '. id: ' + str_json.get('transcription_id') + "\n\t status: " + str_json.get('status'))
            count+=1
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > len(file_line) or user_in < 0:
                raise Exception()
            else:
                break
        except:
            print("Enter a valid option!")
            input("Press Enter to Continue...")
    if user_in == 0:
        transcription_menu()
    elif user_in > 0 and user_in <= len(file_line):
        line_json = json.loads(file_line[user_in - 1])
        files_url = line_json.get('files')
        trans_id = line_json.get('self').split('/')[-1]
        in_transcription_menu(files_url, trans_id)

# ------ specific transciption menu ------ 
def in_transcription_menu(files_url, transcript_id):
    res = ts.batch_transcription_response(files_url)
    result_list = ts.response_intermediate(res, transcript_id)
    while True:
        count = 2
        print("--Request " + transcript_id + " Menu --")
        print("0. Back to Transcription Menu")
        print("1. Print results for all transcriptions")
        for item in result_list:
            print(str(count) + '.')
            print("\tName: " + item.get('name'))
            print("\tType: " + item.get('kind'))
            #print("\tContent URL: " + item.get('links'))
            count+=1
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > len(result_list) + 1 or user_in < 0:
                raise Exception("Out of range")
            elif user_in == 0:
                break
            elif user_in == 1:
                break
            elif user_in <= len(result_list) + 1:
                result_url = result_list[user_in - 2].get('links')
                res_response = ts.batch_transcription_response(result_url)
                res_results = ts.response_results(res_response, transcript_id)
                print("Results")
                for result in res_results:
                    print("\tRecognized: " + result)
                input("Press Enter to Continue...")
            else:
                break
        except Exception as ex:
            print("Enter a valid option!")
            print(ex)
            input("Press Enter to Continue...")
    if user_in == 0:
        transcription_menu()
    elif user_in == 1:
        res_trans = ''
        for item in result_list:
            if item.get('kind') == 'Transcription':
                result_url = item.get('links')
                res_response = ts.batch_transcription_response(result_url)
                res_results = ts.response_results(res_response, transcript_id)
                print("Results for " + item.get('name'))
                res_trans += "Results for " + item.get('name') + "\n"
                for result in res_results:
                    print("\tRecognized: " + result)
                    res_trans += "\tRecognized: " + result + "\n"
        try:
            with open('./responses/' + transcript_id + "/combined_results" + ".txt", "w") as outfile:
                outfile.write(res_trans)
        except:
            print("Cannot open combined_results.txt for writing")
        input("Press Enter to Continue...")
        transcription_menu()

# ------ refrest all transcriptions ------ 
def refresh_transcriptions():
    print("Refreshing Transcription Status...")
    file_line = []
    with open('./transcriptions.txt', "r") as outfile:
        file_line = outfile.readlines()
    os.remove('./transcriptions.txt')
    with open('./transcriptions.txt', "w") as outfile:
        outfile.close()
    for i in file_line:
        str_json = json.loads(i)
        self_url = str_json.get("self")
        res = ts.batch_transcription_response(self_url)
        initial_dict = ts.response_initial(res)
    print("Transcription Statuses Refreshed")
    input("Press Enter to Continue...")
    transcription_menu()

# ------ delete transcription menu ------ 
def transcription_delete_menu():
    while True:
        print("Select Transciption for Deletion: ")
        print("0. Back to Transciption Menu")
        file_line = []
        count = 1
        with open('./transcriptions.txt', "r") as outfile:
            file_line = outfile.readlines()
        for i in file_line:
            str_json = json.loads(i)
            print(str(count) + '. id: ' + str_json.get('transcription_id') + "\n\t status: " + str_json.get('status'))
            count+=1
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > len(file_line) or user_in < 0:
                raise Exception()
            else:
                break
        except:
            print("Enter a valid option!")
            input("Press Enter to Continue...")
    if user_in == 0:
        transcription_menu()
    elif user_in > 0 and user_in <= len(file_line):
        trans_url = json.loads(file_line[user_in - 1]).get('self')
        ts.delete_batch_transcription(trans_url)
        input("Press Enter to Continue...")
        transcription_menu()

# ------ azure storage menu ------ 
def storage_menu():
    while True:
        print("--Storage Menu--")
        print("0. Back to Main Menu\n1. Select Containers\n2. Create new Container\n3. Delete Container")
        user_in = input("Select Option: ")
        try:
            user_in = int(user_in)
            if user_in > 3 or user_in < 0:
                raise Exception()
            else:
                break
        except:
            print("Enter a valid option!")
            input("Press Enter to Continue...")
    if user_in == 0:
        main_menu()
    elif user_in == 1:
        view_container_menu()
    elif user_in == 2:
        create_container_menu()
    elif user_in == 3:
        delete_container_menu()

# ------ azure storage container menu ------ 
def view_container_menu():
    while True:
        print("--Container Menu--")
        print("Available Containers:")
        name_list = bu.list_containers(blob_service_client)
        print("0. Back to Storage Menu")
        user_in = input("Select Option: ")
        try: 
            user_in = int(user_in)
            if user_in > len(name_list) or user_in < 0:
                raise Exception()
            else: 
                break
        except:
            print("Enter a valid option!")
            input("Press Enter to Continue...")
    if user_in == 0:
        storage_menu()
    else:
        container_name = name_list[user_in - 1]
        in_container_menu(container_name)

# ------ specific azure container menu ------ 
def in_container_menu(container_name):
    while True:
        print("--Container " + container_name + " Menu--")
        print("0. Back to Storage Menu\n1. View Blobs\n2. Upload file to Container")
        user_in = input("Select Option: ")
        try: 
            user_in = int(user_in)
            if user_in > 2 or user_in < 0:
                raise Exception()
            elif user_in == 0: 
                break
            elif user_in == 1:
                bu.list_blobs(blob_service_client, container_name)
            elif user_in == 2:
                print("If path is file, upload file to azure\nIf path is folder, upload all files in folder to azure\nType 'x' to cancel")
                filepath = input("Enter path: ")
                if filepath == "x":
                    pass
                elif os.path.isfile(filepath):
                    bu.upload_blob_to_container(blob_service_client, container_name, filepath)
                    input("Press Enter to Continue...")
                elif os.path.isdir(filepath):
                    folder_contents = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
                    for f in folder_contents:
                        bu.upload_blob_to_container(blob_service_client, container_name, os.path.join(filepath, f))
                    input("Press Enter to Continue...")
                else:
                    print("Not a file nor folder, try again.")
                    input("Press Enter to Continue...")
        except Exception as ex:
            print(ex)
            print("Enter a valid option!")
            input("Press Enter to Continue...")
    if user_in == 0:
        storage_menu()

# ------ create new container menu ------ 
def create_container_menu():
    print("--Create New Container-")
    print("Type 'x' to cancel")
    user_in = input("Enter Container Name: ")
    if user_in == 'x':
        pass
    else:
        bu.create_container(blob_service_client, user_in)
        input("Press Enter to Continue...")
    storage_menu()

# ------ delete container menu ------ 
def delete_container_menu():
    while True:
        print("--Delete Container Menu--")
        print("Available Containers:")
        name_list = bu.list_containers(blob_service_client)
        print("0. Back to Storage Menu")
        user_in = input("Select Container: ")
        try: 
            user_in = int(user_in)
            if user_in > len(name_list) or user_in < 0:
                raise Exception()
            else: 
                break
        except:
            print("Enter a valid option!")
            input("Press Enter to Continue...")
    if user_in == 0:
        storage_menu()
    elif user_in <= len(name_list) and user_in > 0:
        bu.delete_container(blob_service_client, name_list[user_in - 1])
        input("Press Enter to Continue...")
        storage_menu()

if __name__ == "__main__":
    #ts.video_to_wav('./test4.mkv')
    main_main_menu()
    #res = ts.batch_transcription_response("https://eastasia.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions/70b6a040-2cc3-44ed-a54d-8ef244f0cfb6/files")
    #print(res.json())
    #print(res.status_code)