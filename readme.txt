Github Repo for Team NF ict 2202

Installation steps

ffmpeg	https://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/

pip install pydub	
pip install azure-storage-blob azure-identity
pip install azure-cognitiveservices-speech
pip install moviepy
pip install tensorflow
pip install pickle-mixin
pip install numpy
pip install keras (do additional 'pip install keras --upgrade')
pip install opencv-python
pip install image-similarity-measures

# To Access Azure Resource use 'setx' command in cmd 
setx SPEECH_KEY <speech_key> 
setx AZURE_STORAGE_CONNECTION_STRING <connection_string>

Due to Security advice from Azure, resources to access Azure services will be emailed to Professor Wei Han