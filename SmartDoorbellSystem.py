import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import pickle
import csv
from monitor_upload import load_data
import os


USING_RPI_CAMERA_MODULE = False

known_face_encodings = []
known_face_metadata = []

known_face_name_owner = []
owner_face_encodings = []

facial_encodings_folder = 'data_SmartDoorbell/'

def load_facial_encodings_and_names_from_memory():
    for filename in os.listdir(facial_encodings_folder):
        known_face_name_owner.append(filename[:-4])
        with open(facial_encodings_folder + filename, 'rb') as fp:
            owner_face_encodings.append(pickle.load(fp)[0])
            print("da load chu nha")


def save_known_faces():
    with open("data/faces.pickle", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Known faces backed up to disk.")


def load_known_faces():
    global known_face_encodings, known_face_metadata

    try:
        with open("data/faces.pickle", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass


def register_new_face(face_encoding, face_image):
    known_face_encodings.append(face_encoding)

    known_face_metadata.append({
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
    })
    label_info_faces = ['face_image', 'first_seen', 'last_seen', 'first_seen_this_interaction', 'seen_count']
    data = [[], datetime.now(), datetime.now(),
            datetime.now(), 0]
    os.makedirs(os.path.dirname("data/info_faces.cvs"), exist_ok=True)
    with open("data/info_faces.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(label_info_faces)
        writer.writerow(data)

def lookup_known_face(face_encoding):
    metadata = None

    if len(known_face_encodings) == 0:
        return metadata

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] < 0.65:

        metadata = known_face_metadata[best_match_index]

        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=1):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1
    return metadata



def main_loop():
    video_capture = cv2.VideoCapture(0)

    number_of_faces_since_save = 0

    while True:

        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0 , 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_labels = []
        face_names_owner = []

        for face_encoding_owner in face_encodings:

            matches_owner = face_recognition.compare_faces(owner_face_encodings, face_encoding_owner)
            face_distances_owner = face_recognition.face_distance(owner_face_encodings, face_encoding_owner)
            best_match_index_owner = np.argmin(face_distances_owner)

            if matches_owner[best_match_index_owner]:
                name_owner = known_face_name_owner[best_match_index_owner]

                face_names_owner.append(name_owner)
            print(face_names_owner)

        for face_location, face_encoding in zip(face_locations, face_encodings):

            metadata = lookup_known_face(face_encoding)

            if metadata is not None:

                time_at_door = datetime.now() - metadata['first_seen_this_interaction']

                label_info_faces = ['face_image', 'first_seen', 'last_seen', 'first_seen_this_interaction',
                                    'seen_count']

                data = [metadata["face_image"], metadata["first_seen"], metadata["last_seen"],
                        metadata['first_seen_this_interaction'], metadata['seen_count']]

                with open('data/info_faces.csv', 'w', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(label_info_faces)
                    writer.writerow(data)

                face_label = f"At door {int(time_at_door.total_seconds())}s"
                start = datetime.now()

                time_notification = (time_at_door.total_seconds())
                label = start.strftime("%Y-%m-%d_%H-%M-%S")

                if 5 < time_notification < 5.1 and face_names_owner != ['owner']:
                    cv2.imwrite('image/' + str(label) + "_image.jpg", frame)
                    load_data()
            else:
                face_label = "New visitor!"
                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))

                register_new_face(face_encoding, face_image)

            face_labels.append(face_label)

        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        number_of_recent_visitors = 0

        for metadata in known_face_metadata:

            if datetime.now() - metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 1:

                x_position = number_of_recent_visitors * 150
                frame[30:180, x_position:x_position + 150] = metadata["face_image"]
                number_of_recent_visitors += 1

                visits = metadata['seen_count']
                visit_label = f"{visits} visits"
                if visits == 1:
                    visit_label = "First visit"
                cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255),
                            1)

        if number_of_recent_visitors > 0:
            cv2.putText(frame, "Visitors at Door", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('SmartDoorbellSystem', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_facial_encodings_and_names_from_memory()
    load_known_faces()
    main_loop()
