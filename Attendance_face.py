import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import mysql.connector
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import csv

db = mysql.connector.connect(host='smartdoorbellsystem.cynisqrgpez0.ap-southeast-1.rds.amazonaws.com',
                             user='kienlu',
                             password='12345678',
                             database='Attendance_faces', )

scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("attendance-faces-e054fb69a2a3.json", scope)
client = gspread.authorize(creds)

rng = "A2:A"
spreadsheetId = "1mLw0aypxbVeLJmK1-kPxFULHlQQetzH6WCrnLt6_sJk"
sheetName = "Attendance Faces"
spreadsheet = client.open_by_key(spreadsheetId)

worksheet = spreadsheet.worksheet(sheetName)

data = worksheet.get_all_records()
values = worksheet.get(rng)
max_intime = '08:00:00'

video_capture = cv2.VideoCapture(0)

path = 'image_Attendance/'

images = []
ID_User_from_path = []
name_from_path = []
known_face_metadata = []

mylist = os.listdir(path)
for name_ in mylist:
    current_images = cv2.imread(f'{path}/{name_}')
    images.append(current_images)
    name_from_path.append(os.path.splitext(name_)[0].split('_')[0])
    ID_User_from_path.append(os.path.split(name_)[-1].split('_')[-1].split('.')[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList


known_face_encodings = findEncodings(images)


def main_loop():
    count_save = 0
    process_this_frame = True
    face_locations = []
    face_names = []
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                face_names = []
                ID_User_ = []
                if matches[best_match_index]:
                    name = name_from_path[best_match_index]
                    ID_User = ID_User_from_path[best_match_index]
                    ID_User_.append(ID_User)
                    face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Attendance_faces', frame)
        if (len(face_names) != 0) and len(face_locations) > 0 and count_save > 100:
            count_save = 0

            nrows = len(worksheet.col_values(1))
            worksheet.update_cell(nrows + 1, 1, name)
            worksheet.update_cell(nrows + 1, 2, ID_User)
            worksheet.duplicate()

            now = datetime.now()
            date = now.strftime('%m/%d/%Y').replace('/0', '/')
            if (date[0] == '0'):
                date = date[1:]
            time = now.strftime('%H:%M:%S')
            namecell = worksheet.find(name)
            datecell = worksheet.find(date)

            if (time < max_intime):
                worksheet.update_cell(namecell.row, datecell.col, 'present')
                print('recorded')
            else:
                worksheet.update_cell(namecell.row, datecell.col, 'late')
            check_in_hour = datetime.now()
            check_in_day = check_in_hour.strftime("%Y-%m-%d")
            mycursor = db.cursor()
            mycursor.execute(
                "INSERT INTO Attendance_faces(id, FullName, Hour_check_in,Day_check_in) VALUES(%s, %s, %s, %s)",
                (ID_User, name, check_in_hour, check_in_day))
            db.commit()

            header = ['id', 'FullName', 'Hour_check_in', 'Day_check_in']
            data = [ID_User, name, check_in_hour, check_in_day]

            with open('data/data_Attendance', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)

                writer.writerow(header)

                writer.writerow(data)
        else:
            count_save += 1

        keyCode = cv2.waitKey(1)
        if cv2.getWindowProperty("Attendance_faces", cv2.WND_PROP_VISIBLE) < 1:
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
