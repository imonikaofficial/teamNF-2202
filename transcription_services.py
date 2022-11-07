import azure.cognitiveservices.speech as speechsdk
import pydub
import os
import subprocess
import moviepy.editor as mp
import requests as req
import json

speech_key, service_region = "ccfa160ad2634a86aa2063295bef8424","eastasia"
conversion_folder = './converted_audio/'

# unused function 
def from_file(abs_filepath): 
    filename = os.path.basename(abs_filepath)
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_input = speechsdk.AudioConfig(filename=filename)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    result = speech_recognizer.recognize_once_async().get()
    print(result)

# convert m4a audio file to wav audio file
def m4a_to_wav(abs_filepath):
    filepame = os.path.basename(abs_filepath)
    filename = os.path.splitext(filepath)[0]
    pydub.AudioSegment.from_file(abs_filepath + '.m4a').export(conversion_folder + filename + '.wav', format='wav')
    if os.path.exists(conversion_folder + filename + '.wav'):
        print("Converting Success")
    else:
        print("Converting Failure")

# unused function
def video_to_wav_test(abs_filepath):
    filename = os.path.basename(abs_filepath)
    filename = os.path.splitext(filepath)[0]
    command = "ffmpeg -i " + filepath + "-vn -acodec copy" + './tmp/' + filename + '.wav'
    if command: 
        print("success")
    else:
        print("fail")

# convert video files that can be processed by moviepy to wav audio
def video_to_wav(abs_filepath):
    filepath = os.path.basename(abs_filepath)
    filename = os.path.splitext(filepath)[0]
    video_clip = mp.VideoFileClip(r"" + abs_filepath)
    video_clip.audio.write_audiofile(r"" + conversion_folder + filename + '.wav')
    if os.path.exists(conversion_folder + filename + '.wav'):
        print("Converting Success")
    else:
        print("Converting Failure")

# function to create batch transcription, returns REST response from azure
def create_batch_transription(transcriptionURI, requestSendURI, transcriptionName="Test Transcription"):
    print("Creating Transciption Request...")
    header = {'Content-Type': 'application/json', 
    'Ocp-Apim-Subscription-Key': f'{speech_key}'}
    json_body = r'{"contentContainerUrl": "' + transcriptionURI + r'", "locale": "en-US", "displayName": "' + transcriptionName + r'", "model": null, "properties": {"wordLevelTimestampsEnabled": true,},}'
    response = req.post(requestSendURI, data=json_body, headers=header, verify=False)
    if str(response.status_code) == '201':
        print("Transcription request successfully created")
    else:
        print("ERROR : Transciption not created")
    return response

# function to get response from url, returns REST response from azure
def batch_transcription_response(response_url):
    header = {'Ocp-Apim-Subscription-Key': f'{speech_key}'}
    response = req.get(headers=header, url=response_url)
    # TODO : Status Code detection
    return response

# function to delete batch transcription , returns status code 204 if successful, else REST response from azure
def delete_batch_transcription(transcript_url):
    header = {'Ocp-Apim-Subscription-Key': f'{speech_key}'}
    response = req.delete(transcript_url, headers=header)
    transcript_id = transcript_url.split('/')[-1]
    if str(response.status_code) == '204':
        print("The transcription task was successfully deleted from azure.")
        file_lines = []
        check_hit = 0
        hit_line = ''
        with open('./transcriptions.txt', "r") as outfile:
            file_lines = outfile.readlines()
        for each_line in file_lines:
            if str(transcript_id) in each_line:
                check_hit = 1
                hit_line = each_line
                break
        if check_hit == 1:
            file_lines.remove(hit_line)
        with open('./transcriptions.txt', "w") as outfile:
            for item in file_lines:
                outfile.write(item)
            print("Deleted transcription ID from transcriptions.txt")
    else:
        print("ERROR : Transcription task failed to delete")
    return response

# function to read initial response from azure
def response_initial(response):
    res_json = json.dumps(response.json(), indent=4)
    res_load = json.loads(res_json)
    # getting useful info from response
    files_link = res_load.get('links').get('files')
    req_status = res_load.get('status')
    self_link = res_load.get('self')
    self_list = self_link.split('/')
    self_transcription = self_list[-1]
    # making folder for response based off transcription id
    try: 
        if not os.path.exists('./responses/' + self_transcription):
            os.mkdir('./responses/' + self_transcription)
    except:
        print("ERROR : Can not create folder for transcription.")
    # creating initial response json file inside folder
    try: 
        with open('./responses/' + self_transcription + '/1_initial_' + self_transcription + '.json', "w") as outfile:
                outfile.write(res_json)
    except: 
        print("ERROR : Can not create file for initial json.")
    # entering transcription id into file ./transcription.txt
    save_transcription_id('{"files" : "'+files_link+'", "status" : "'+req_status+'", "self" : "'+self_link+'", "transcription_id" : "'+self_transcription+'"}' + "\n", self_transcription)
    # creating variable for return
    return_dict = {'files' : files_link, 'status' : req_status, 'self' : self_link, 'transcription_id' : self_transcription} 
    return return_dict

# function to read response from azure when status=success
def response_intermediate(response, transcription_id):
    res_json = json.dumps(response.json(), indent=4)
    # creating intermediate response json file inside folder
    try: 
        with open('./responses/' + transcription_id + '/2_intermediate_' + transcription_id + '.json', "w") as outfile:
                outfile.write(res_json)
    except: 
        print("ERROR : Can not create file for initial json.")
    # getting useful information for response
    list_results = json.loads(res_json).get('values')
    result_list = []
    for item in list_results:
        result_url = item.get('self')
        result_kind = item.get('kind')
        if str(result_kind) == 'Transcription':
            result_name = item.get('name').split('/')[-1]
        else:
            result_name = item.get('name')
        result_link = item.get('links').get('contentUrl')
        result_list.append({'self' : result_url, 'name' : result_name, 'kind': result_kind, 'links' : result_link})
    return result_list

# function to get audio file transcription results from azure
def response_results(response, transcription_id):
    res_json = json.dumps(response.json(), indent=4)
    res_load = json.loads(res_json)
    file_name = res_load.get('source').split('/')[-1]
    # creating result file inside folder for specific audio file
    with open('./responses/' + transcription_id + '/3_' + file_name + '.json', "w") as outfile:
        outfile.write(res_json)
    res_list = res_load.get('combinedRecognizedPhrases')
    results = []
    for result in res_list: 
        combined = result.get('display')
        results.append(combined)
    # TODO : Output
    return results

# saving transcription id to text file
def save_transcription_id(response_string, transcription_id):
    check_hit = 0
    file_lines = []
    try:
        with open('./transcriptions.txt', "r") as outfile:
            file_lines = outfile.readlines()
        for each_line in file_lines:
            if str(transcription_id) in each_line:
                check_hit = 1
                print("Transciption ID already saved")
                break
        if check_hit == 0:
            file_lines.append(response_string)
            print("Saving Transcription ID")
        with open('./transcriptions.txt', "w") as outfile:
            for item in file_lines:
                outfile.write(item)
    except:
        print("Can not open transcriptions.txt")

def test(transcript_id):
    file_lines = []
    with open('./transcriptions.txt', "r") as outfile:
        file_lines = outfile.readlines()
    print(file_lines)
    if str(transcript_id) + '\n' in file_lines:
        file_lines.remove(str(transcript_id) + '\n')
    with open('./transcriptions.txt', "w") as outfile:
        for item in file_lines:
            outfile.write(item)


"""
if __name__ == "__main__":
    #video_to_wav("./test3.mkv")
    #m4a_to_wav("./test.m4a")
    #from_file("./tmp/test.wav")

    #transcription file location URI format : "https://<storage_account_name>.blob.core.windows.net/<container_name>"
    #transcription request send location format : "https://<YourServiceRegion>.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions"
    target_container = "https://02projectstorage.blob.core.windows.net/testcontainer02"

    #test1 = create_batch_transription(target_container, "https://" + service_region + ".api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions")
    #test1 = batch_transcription_response('https://eastasia.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions/0ee45a0e-e587-4b5b-9658-523f44f17f72')
    #test1 = batch_transcription_response(r'https://spsvcprodae.blob.core.windows.net/bestor-1ac74883-7268-4fe4-a91c-d7be81185064/TranscriptionData/0ee45a0e-e587-4b5b-9658-523f44f17f72_0_0.json?sv=2021-08-06&st=2022-10-31T16%3A25%3A42Z&se=2022-11-01T04%3A30%3A42Z&sr=b&sp=rl&sig=phMJPWD6F%2FcY9LDFMpHTx5pc4XiVkFLyRwyEK%2Flxu%2Fs%3D')
    
    #test1 = batch_transcription_response('https://eastasia.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions/4aab527c-8c02-4a31-afbc-8a304a213cc0')
    #test1 = batch_transcription_response('https://spsvcprodae.blob.core.windows.net/bestor-1ac74883-7268-4fe4-a91c-d7be81185064/TranscriptionData/4aab527c-8c02-4a31-afbc-8a304a213cc0_0_2.json?sv=2021-08-06&st=2022-11-01T13%3A12%3A49Z&se=2022-11-02T01%3A17%3A49Z&sr=b&sp=rl&sig=Z0jhSz%2FlXqIjT38mVCpo5uQDAU%2FBopmwA3OFW575Q6I%3D')
    #test1 = delete_batch_transcription('https://eastasia.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions/0ee45a0e-e587-4b5b-9658-523f44f17f72')

    #print("JSON RAW: " + str(json.dumps(test1.json(), indent=4)))
    #print("STATUS CODE RAW: " + str(test1.status_code))

    #print_response_files(test1)
    #test_read_inital(test1) done
    #test_results(test1)

    #print(response_initial(test1))
    #print(response_intermediate(test1, '4aab527c-8c02-4a31-afbc-8a304a213cc0'))
    #print(response_results(test1, '4aab527c-8c02-4a31-afbc-8a304a213cc0'))
    #delete_batch_transcription('https://eastasia.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions/4aab527c-8c02-4a31-afbc-8a304a213cc0')
    #delete_batch_transcription('https://eastasia.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions/4aab527c-8c02-4a31-afbc-8a304a213cc0')

    while True:
        print("--2202 Project Transciption--")
        print("0. Exit Program\n1. Azure Storage Services\n2. Transcribe Files in Azure Storage")
        user_in = input("Select Option: ")
        print(type(user_in))
        if user_in == '1':
            print("0.")
    # TODO : put everything together into program with user input
"""