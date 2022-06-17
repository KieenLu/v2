import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import pickle
import csv
import os
from monitor_upload import load_data
# Set this depending on your camera type:
# - True = Raspberry Pi 2.x camera module
# - False = USB webcam or other USB video input (like an HDMI capture device)
USING_RPI_CAMERA_MODULE = False
facial_encodings_folder = 'data_SmartDoorbell/'
# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []
known_face_names_owner = []
known_face_encodings_owner = []

def load_facial_encodings_and_names_from_memory():
    for filename in os.listdir(facial_encodings_folder):
        known_face_names_owner.append(filename[:-4])
        with open(facial_encodings_folder + filename, 'rb') as fp:
            known_face_encodings_owner.append(pickle.load(fp)[0])

def save_known_faces():
    with open("data/known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def load_known_faces():
    global known_face_encodings, known_face_metadata

    try:
        with open("data/known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass




def register_new_face(face_encoding, face_image):
    """
    Add a new person to our list of known faces
    """
    # Add the face encoding to the list of known faces
    known_face_encodings.append(face_encoding)
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
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
    """
    See if this is a face we already have in our face list
    """
    metadata = None

    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0:
        return metadata

    # Calculate the face distance between the unknown face and every face on in our known face list
    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
    # the more similar that face was to the unknown face.
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = np.argmin(face_distances)

    # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
    # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
    # of the same person always were less than 0.6 away from each other.
    # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
    # people will come up to the door at the same time.
    if face_distances[best_match_index] < 0.65:
        # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
        metadata = known_face_metadata[best_match_index]

        # Update the metadata for the face so we can keep track of how recently we have seen this face.
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

        # We'll also keep a total "seen count" that tracks how many times this person has come to the door.
        # But we can say that if we have seen this person within the last 5 minutes, it is still the same
        # visit, not a new visit. But if they go away for awhile and come back, that is a new visit.
        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=1):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1

    return metadata


def main_loop():

    video_capture = cv2.VideoCapture(1)

    # Track how long since we last saved a copy of our known faces to disk as a backup.
    number_of_faces_since_save = 0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding_owner in face_encodings:
            results = face_recognition.compare_faces(known_face_encodings_owner, face_encoding_owner)
            print(results)
            break

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):

            metadata = lookup_known_face(face_encoding)
            # If we found the face, label the face with some useful information.
            if metadata is not None:
                time_at_door = datetime.now() - metadata['first_seen_this_interaction']

                face_label = f"At door {int(time_at_door.total_seconds())}s"
                label_info_faces = ['face_image', 'first_seen', 'last_seen', 'first_seen_this_interaction',
                                    'seen_count']

                data = [metadata["face_image"], metadata["first_seen"], metadata["last_seen"],
                        metadata['first_seen_this_interaction'], metadata['seen_count']]

                with open('data/info_faces.csv', 'w', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(label_info_faces)
                    writer.writerow(data)

                start = datetime.now()

                time_notification = (time_at_door.total_seconds())
                label = start.strftime("%Y-%m-%d_%H-%M-%S")

                if 5 < time_notification < 5.1 and results == [False]:
                    cv2.imwrite('image/' + str(label) + "_image.jpg", frame)
                    load_data()
            # If this is a brand new face, add it to our list of known faces
            else:
                face_label = "New visitor!"

                # Grab the image of the the face from the current frame of video
                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))

                # Add the new face to our known face data
                register_new_face(face_encoding, face_image)

            face_labels.append(face_label)

        # Draw a box around each face and label each face
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display recent visitor images
        number_of_recent_visitors = 0
        for metadata in known_face_metadata:
            # If we have seen this person in the last minute, draw their image
            if datetime.now() - metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 1:
                # Draw the known face image
                x_position = number_of_recent_visitors * 150
                frame[30:180, x_position:x_position + 150] = metadata["face_image"]
                number_of_recent_visitors += 1

                # Label the image with how many times they have visited
                visits = metadata['seen_count']
                visit_label = f"{visits} visits"
                if visits == 1:
                    visit_label = "First visit"
                cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        if number_of_recent_visitors > 0:
            cv2.putText(frame, "Visitors at Door", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display the final frame of video with boxes drawn around each detected fames
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break

        # We need to save our known faces back to disk every so often in case something crashes.
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_facial_encodings_and_names_from_memory()
    load_known_faces()
    main_loop()
