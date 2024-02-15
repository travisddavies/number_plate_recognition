# Number Plate Recognition
## Introduction
This project aims to accurately detect number plates on cars and extract the
number plate from an image/video.

<p align="center">
  <img src="assets/result.gif" alt="Alt Text">
</p>

## Note
This model is specific to Chinese number plates. This model will likely not work
well on number plates from other countries.

## In the Works
- Deploying a web interface onto a raspberry pi

## How to Use

#### Clone the Repo and Install Dependencies
```
git clone git@github.com:travisddavies/number_plate_recognition.git

cd number_plate_recognition

pip install -r requirements.txt
```


#### Run the software

For image recognition:

```
python3 image_recognition.py -c <au or ch> -i <input image> -o <saved image>
```

For video recognition:

```
python3 video_recognition.py -c <au or ch> -i <input video> -o <saved video>
```

Example:
```
python3 video_recognition.py -i samples/IMG_0467.MOV -o result.MOV
```

### Deployment onto Raspberry Pis
A yaml script has been written to deploy this model onto a **Raspi OS Bookworm** Raspberry Pi.
This deployment will run a script that will deploy a live recording model that accesses your
Pi Camera and sends the data of passing number plates to a CouchDB database.

To update the information for your particular Raspberry Pi, change the information in hosts.ini
to your information. The username to access the database is "admin" and the password is "password".

#### Note:
A bug exists due a deprecation in the paddleocr code in PaddleOCR/ppocr/postprocess/db_postprocess.py
at line 188 to 191. To fix this, simply change the "np.int" part of each line to "np.int64" (I know this
doesn't sound very professional but unfortunately deployment of PaddleOCR on a Raspberry Pi was not easy).

To deploy the model, simple run the following:

```
ansible-playbook pi_deployment.yaml
```

This should deploy all dependencies, open ports, deploy the docker containers, set up venvs, etc to run
the model on a Raspberry Pi.
