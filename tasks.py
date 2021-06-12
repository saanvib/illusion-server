from celery import Celery
from time import sleep
import subprocess

app = Celery('tasks', broker='amqp://', backend='db+sqlite:///db.sqlite3')

@app.task
def reverse(text):
    sleep(5)
    return text[::-1]

@app.task
def featureextract(filename):
    result = subprocess.run(["/home/rishi_bhargava_gmail_com/OpenFace/build/bin/FeatureExtraction", "-f", "/home/rishi_bhargava_gmail_com/OpenFace/samples/default.wmv", "-out_dir", "/home/rishi_bhargava_gmail_com/Fiddle/temp"])    
    return filename



