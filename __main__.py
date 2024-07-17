import cv2
import mediapipe as mp
import numpy as np
import threading

def calculate_distance(top, bottom):
    top = np.array(top)
    bottom = np.array(bottom)

    dist = np.linalg.norm(top - bottom)
    return dist


mp_face_detection = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_face_detection.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame")
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_detection.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_detection.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_detection.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                try:
                    top_left_eye = [results.multi_face_landmarks[0].landmark[386].x, results.multi_face_landmarks[0].landmark[386].y]
                    bottom_left_eye = [results.multi_face_landmarks[0].landmark[374].x, results.multi_face_landmarks[0].landmark[374].y]
                    top_right_eye = [results.multi_face_landmarks[0].landmark[159].x, results.multi_face_landmarks[0].landmark[159].y]
                    bottom_right_eye = [results.multi_face_landmarks[0].landmark[145].x, results.multi_face_landmarks[0].landmark[145].y]
                    if ((calculate_distance(top_right_eye, bottom_right_eye) and calculate_distance(top_left_eye, bottom_left_eye)) < 0.01 ):
                        print("BLINK!")
                except:
                    pass
                    
        # Renders to the display
        cv2.imshow("Mediapipe Feed", cv2.flip(image, 1))
        # Waits for key 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

