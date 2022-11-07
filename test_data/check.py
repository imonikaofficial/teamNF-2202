import os
import shutil

def pause(counter, condition):
    if counter == condition:
        input("checkpoint")
        counter = 0
    else:
        counter += 1
    return counter

def check_number():
    a = os.listdir("Flicker8k_Dataset")
    b = ["Flickr_8k.devImages.txt","Flickr_8k.testImages.txt","Flickr_8k.trainImages.txt"]
    filestreams = []
    for i in b:
        f = open(i,"r")
        filestreams.append(f.read().split("\n"))
        f.close()
    count = 0

    check = [0,0,0]
    for i in a:
        for j in range(len(filestreams)):
            if i in filestreams[j]:
                check[j] += 1
        #print(i, check)
        #count = pause(count, 20)
    print(check)

def get_small_test_set():
    f = open("Flickr_8k.trainImages.txt","r")
    data = f.read().split("\n")
    f.close()
    f = open("Flickr_8k.testImages.txt","r")
    data2 = f.read().split("\n")
    f.close()

    short_data = data[:50]
    short_test = data2[:50]
    whole_data = short_data + short_test
    try:
        os.mkdir("Flicker8k_Dataset_small")
    except:
        shutil.rmtree("Flicker8k_Dataset_small")
        os.mkdir("Flicker8k_Dataset_small")    
    for i in whole_data:
        filename = "Flicker8k_Dataset_small/" + i
        with open("Flicker8k_Dataset/" + i,"rb") as f:
            f2 = open(filename,"wb+")
            f2.write(f.read())
            f2.close()
    
    f = open("Flickr_8k.trainImages_small.txt","w")
    f.write("\n".join(short_data) + "\n")
    f.close()    
    f = open("Flickr_8k.testImages_small.txt","w")
    f.write("\n".join(short_test) + "\n")
    f.close()

    needed_tokens = []
    with open("Flickr8k.token.txt","r") as f:
        token_data = f.read().split("\n")
        for i in token_data:
            if i[:i.find("#")] in whole_data:
                needed_tokens.append(i)

    with open("Flickr8k.token_small.txt","w+") as f:
        f.write("\n".join(needed_tokens) + "\n")
        
        
get_small_test_set()

