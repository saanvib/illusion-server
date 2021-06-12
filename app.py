from celery import Celery
from celery.result import AsyncResult
from flask import Flask, request
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from pytube import YouTube
import os
 

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery



flask_app = Flask(__name__)
flask_app.config.update(
    CELERY_BROKER_URL='amqp://',
    CELERY_RESULT_BACKEND='db+sqlite:///db.sqlite3'
)
celery = make_celery(flask_app)

# this is the celery task for training the model
# filename would be youtubeid_fake or youtubeid_real
@celery.task(name='trainmodel')
def trainmodel(celebrity, youtubeId, real):
    downloadVid(youtubeId)
    dir_name = "/home/rishi_bhargava_gmail_com/Fiddle/training_videos/" + celebrity
    if (not os.path.isdir(dir_name)):
        os.makedirs(dir_name)
    os.replace("/tmp/" + youtubeId + ".mp4", "/home/rishi_bhargava_gmail_com/Fiddle/training_videos/" + celebrity + "/" + youtubeId + "_" + real + ".mp4")
    filenames = os.listdir(dir_name)
    print(filenames)
    output_dir = "/tmp/" + celebrity
    dfs = []
    for filename in filenames:
        full_filename = dir_name + "/" + filename
        result = subprocess.run(["/home/rishi_bhargava_gmail_com/OpenFace/build/bin/FeatureExtraction", "-f", full_filename, "-out_dir", output_dir])
        extract_dirname = filename.split('.')[0]
        print(extract_dirname)
        csv_file = output_dir + "/" + extract_dirname + ".csv"
        cols = list(pd.read_csv(csv_file, nrows =1))
        # Define unused cols
        unused = ['frame', 'face_id', 'timestamp']
        # Use list comprehension to remove the unwanted column in **usecol**
        df = pd.read_csv(csv_file, usecols =[i for i in cols if i not in unused])
        if "real" in extract_dirname:
            df.insert(0, 'class', 1)
        else:
            df.insert(0, 'class', 0)
        dfs.append(df)
    dataset = pd.concat(dfs)
    
    X = dataset.iloc[:,1:].values
    Y = dataset.iloc[:,:1].values.ravel()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    scaler = StandardScaler()
    # normalization
    #scaler.fit(X)
    scaler.fit(X_train)
    dump(scaler, '/home/rishi_bhargava_gmail_com/Fiddle/models/scalar_' + celebrity + ".joblib")
    #X = scaler.transform(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=5)
    # training the model
    classifier.fit(X_train, Y_train)
    model_file = '/home/rishi_bhargava_gmail_com/Fiddle/models/' + celebrity + ".joblib"
    dump(classifier, model_file)
    y_pred = classifier.predict(X_test)
    print(classification_report(Y_test, y_pred))
    print(confusion_matrix(Y_test, y_pred))
    return

@celery.task(name='classify')
def classify(youtubeId, celebrity):
    celebrity = celebrity.lower()
    downloadVid(youtubeId)
    filename = "/tmp/" + youtubeId + ".mp4"
    #filename = "/home/rishi_bhargava_gmail_com/Fiddle/training_videos/obama/obama1_fake.mp4"
    output_dir = "/tmp/" + youtubeId
    result = subprocess.run(["/home/rishi_bhargava_gmail_com/OpenFace/build/bin/FeatureExtraction", "-f", filename, "-out_dir", output_dir])
    csv_file = output_dir + "/" + youtubeId + ".csv"
    cols = list(pd.read_csv(csv_file, nrows =1))
    # Define unused cols
    unused = ['frame', 'face_id', 'timestamp']
    # Use list comprehension to remove the unwanted column in **usecol**
    # read 2 files without the first 3 columns
    df1 = pd.read_csv(csv_file, usecols =[i for i in cols if i not in unused])
    X = df1.iloc[:,0:].values
    model_file = '/home/rishi_bhargava_gmail_com/Fiddle/models/' + celebrity + ".joblib"
    classifier = load(model_file) 
    scaler = load('/home/rishi_bhargava_gmail_com/Fiddle/models/scalar_' + celebrity + ".joblib")
    X = scaler.transform(X)
    # what to do with normalization
    y_pred = classifier.predict(X)
    print(y_pred)
    count = 0
    for x in y_pred:
        if x == 1:
            count+=1
    print(" count of 1s in y_pred: " + str(count))
    print("length : " + str(len(y_pred)))
    val1 = round((len(y_pred)-count)/len(y_pred), 2)
    val2 = round(count/len(y_pred), 2)
    print(val1)
    print(val2)
    if (count < .7*len(y_pred)):
        return "False "+str(val1)
    return "True " + str(val2)

@flask_app.route('/')
def test():
    return 'hello'

@flask_app.route('/classify')
def classify_request():
    celebrity = request.args.get('celebrity')
    youtubeId = request.args.get('youtubeid')
    res = classify.delay(youtubeId, celebrity)
    return res.id

# taking a youtube id, celebrity name, and training the model for that celebrity
@flask_app.route('/train')
def train_model():
    celebrity = request.args.get('celebrity').lower()
    youtubeId = request.args.get('youtubeid')
    real = request.args.get('category')
    print("celeb " + celebrity)
    print("youtube " + youtubeId)
    print("authenticity " + real)
    res = trainmodel.delay(celebrity, youtubeId, real)
    return res.id

@flask_app.route('/status/<jobid>')
def getStatus(jobid):
    res = AsyncResult(jobid, app=celery)
    if res.status == "SUCCESS":
        return str(res.get())
    else:
        return res.status

def downloadVid(youtubeId):
    SAVE_PATH = "/tmp" #to_do 
    # link of the video to be downloaded 
    link="https://www.youtube.com/watch?v=" + youtubeId
    
    try: 
        # object creation using YouTube
        # which was imported in the beginning 
        yt = YouTube(link) 
    except: 
        print("Connection Error") #to handle exception 
    
    # filters out all the files with "mp4" extension 

    
    # mp4files = yt.filter('mp4') 
    
    # #to set the name of the file
    # yt.set_filename(youtubeId)  
    
    # # get the video with the extension and resolution passed in the get() function 
    # d_video = yt.get(mp4files[-1].extension,mp4files[-1].resolution) 
    try: 
        # downloading the video 
        # d_video.download(SAVE_PATH) 
        yt.streams.filter(progressive = True, file_extension = "mp4").get_highest_resolution().download(output_path = SAVE_PATH, filename = youtubeId)

    except: 
        print("Some Error!") 
    print('Task Completed!') 

if __name__ == '__main__':
    flask_app.run(port=5000,host='0.0.0.0',debug=True)
