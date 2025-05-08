from flask import Flask
import threading
import cv2
import face_recognition

app = Flask(__name__)

def start_face_recognition():
    # --- Your face recognition code ---
    known_face_encodings = []
    known_face_names = []

    def load_and_encode(image_path, name):
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✅ Face loaded for {name}")
            else:
                print(f"⚠️ No face found in {name}'s image!")
        except Exception as e:
            print(f"❌ Error loading {name}'s image: {e}")

    # Load your faces
    load_and_encode(r"F:\final project\face\facer\Images\pp.jpg", "Tuhin")
    load_and_encode(r"F:\final project\face\facer\Images\sharukh.jpg", "Sharukh")
    load_and_encode(r"F:\final project\face\facer\Images\suman.jpeg", "suman")

    if not known_face_encodings:
        print("❌ No known faces loaded. Exiting...")
        return

    # Start webcam
    video_capture = cv2.VideoCapture(0)
    process_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_frame = not process_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

@app.route('/start', methods=['GET'])
def start_task():
    threading.Thread(target=start_face_recognition).start()
    return "Face recognition started!"

if __name__ == '__main__':
    app.run(debug=True)
