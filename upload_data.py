import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import smtplib
from email.message import EmailMessage
import requests
from firebase import Firebase
import pyrebase
import datetime
import pandas as pd
from twilio.rest import Client
import keys

client = Client(keys.acc_sid, keys.auth_token)

Email = 'smartdjangosystem@gmail.com'
Pass = 'lutrungkien'
msg = EmailMessage()
msg['Subject'] = 'SmartDoorbellSystem Notification'
msg['From'] = Email
msg['To'] = 'lukiencm@gmail.com'
msg.set_content("The system detects some anomalies in front of your camera.")
#

config = {
    "apiKey": "AIzaSyBAO8n6tckNFK_2d2ZCHF_cBsY65QN9KHU",
    "authDomain": "smartdoorbellsystem-19df7.firebaseapp.com",
    "databaseURL": "https://smartdoorbellsystem-19df7-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "smartdoorbellsystem-19df7",
    "storageBucket": "smartdoorbellsystem-19df7.appspot.com",
    "messagingSenderId": "616183915042",
    "appId": "1:616183915042:web:8d990d0c0c5d92514de07b",
    "measurementId": "G-FZT0QFSQD1"
}

device_id = '5cfbe20f-1ee9-40d7-a9b3-4624a787e31a'

cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'smartdoorbellsystem-19df7.appspot.com'
})

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
firebase = Firebase(config)


def upload_file(fileName):
    # current_time =datetime.date()
    # put image to firebase
    storage.child(fileName).put(fileName)

    auth = firebase.auth()
    email = "smartdjangosystem@gmail.com"
    password = "lutrungkien"
    user = auth.sign_in_with_email_and_password(email, password)

    url_image = storage.child(fileName).get_url(user['idToken'])

    url = f'http://13.215.178.91:4000/api/v1/devices/{device_id}/notifications'

    df = pd.read_csv('data/info_faces.csv')

    for idx, data in df.iterrows():
        payload = {
            "title" : "SmartDoorbellSystem phát hiện một vị khách dừng lại quá lâu trước Camera của bạn,\n Thông báo ngày : " + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            "description": "Lần đầu tiên vị khách ghé thăm : "+ str(data['first_seen']) + "Lần cuối cùng Camera ghi lại được : " +str(data['last_seen']) + "Tổng số lần mà vị khách này đã ghé thăm : " + str(data['seen_count']) ,
            "imageUrl": url_image
        }
        requests.post(url, json=payload)

    with open(fileName, 'rb') as f:
        file_data = f.read()
        file_name = f.name

    msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Email, Pass)
        smtp.send_message(msg)

    message = client.messages.create(
        body="Đây là tin nhắn cảnh báo từ SmartDoorbellSystem !"
             "\nPhát hiện bất thường trước camera của bạn"
             "\nHãy kiểm tra email của bạn hoặc có thể đăng nhập vào http://13.215.178.91:4000/dashboard/notifications để xem chi tiết về sự bất thường này.",
        from_ = keys.twilio_number,
        to= keys.my_phone_number
    )