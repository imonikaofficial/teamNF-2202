import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def create_container(blobServiceClient, container_name):
    try:
        blob_service_client = blobServiceClient
        container_client = blob_service_client.create_container(container_name)
        print("Container " + container_name + " successfully created.")
    except Exception as ex:
        print("Exception: ")
        print(ex)

def upload_blob_to_container(blobServiceClient, container_name, filepath):
    if os.path.exists(filepath):
        pass # do nothin if exist
    else: 
        exit()
    filename = os.path.basename(filepath)
    blob_service_client = blobServiceClient
    print("Creating blob with name " + filename)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
    print("Uploading data to blob")
    with open(file=filepath, mode="rb") as data:
        blob_client.upload_blob(data)
    print("Uploading to Azure Storage as blob:\n\t" + filepath)


def list_containers(blobServiceClient):
    blob_service_client = blobServiceClient
    container_list = blob_service_client.list_containers()
    #print ("Listing Containers")
    count = 1
    name_list = []
    for c in container_list:
        print ("\t" + str(count) + '. ' + c.name)
        name_list.append(c.name)
        count+=1
    return name_list

def list_blobs(blobServiceClient, container_name):
    blob_service_client = blobServiceClient
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs()
        print("Listing Blobs in " + container_name)
        for blob in blob_list:
            print("\t" + blob.name)
    except Exception as ex:
        print('Exception:')
        print(ex)

def delete_container(blobServiceClient, container_name):
    blob_service_client = blobServiceClient
    try:
        container_client = blob_service_client.get_container_client(container_name)
        container_client.delete_container()
        print("Container " + container_name + " successfully deleted.")
    except Exception as ex:
        print('Exception:')
        print(ex)    


"""
if __name__ == "__main__":
    try:
        print("Azure Blob Storage Python...")

        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        #print(connect_str)

        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        print ("Connection to service")
        list_containers(blob_service_client)
        list_blobs(blob_service_client, "testcontainer02")
        upload_blob_to_container(blob_service_client, "testcontainer02", './tmp/test2.wav')
        upload_blob_to_container(blob_service_client, "testcontainer02", './tmp/test3.wav')

    except Exception as ex:
        print('Exception:')
        print(ex)
"""