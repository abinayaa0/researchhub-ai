FROM python:3.11-slim 
#starts from an existing docker image that already contains a minimal linux env and python 3.11
#slim is a  debian-based image.


WORKDIR /app 
#workdir creates the /app directory if it doesnt exist and makes it the current working directory.
#the below commands are related to the app.
COPY app/requirements.txt .
#copies from local machine into docker image the file requirements.txt
#  . means current working directory inside the image 

RUN pip install --no-cache-dir -r requirements.txt
# installs all the libraries form requirements.txt
#no-cache-dir means dont keep pips download cache after installing packages.
COPY app/ . 
#This copies everything inside the local app/ folder into /app inside the image

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
#specifies the default command to execute
#default command CMD specifies the default command tht runs when a container is started from this image.


#docker file tells how to create an image.
