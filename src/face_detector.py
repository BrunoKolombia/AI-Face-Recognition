import cv2
import numpy as np
import mediapipe as mp
import face_recognition
from typing import List, Tuple, Optional


class FaceDetector:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_faces(self, image: np.ndarray) -> Tuple[List, List]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_image, model="hog")

        results = self.face_mesh.process(rgb_image)

        return face_locations, results

    def get_face_encoding(self, image: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb_image, [face_location])

        if encodings:
            return encodings[0]
        return None

    def draw_face_boxes(self, image: np.ndarray, face_locations: List) -> np.ndarray:
        output = image.copy()

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(output, (left, top), (right, bottom),
                        (0, 255, 0), 2)

        return output

    def draw_face_landmarks(self, image: np.ndarray, face_mesh_results) -> np.ndarray:
        output = image.copy()

        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        return output

    def crop_face(self, image: np.ndarray, face_location: Tuple, padding: int = 20) -> np.ndarray:
        top, right, bottom, left = face_location

        height, width = image.shape[:2]

        top = max(0, top - padding)

        bottom = min(height, bottom + padding)

        left = max(0, left - padding)

        right = min(width, right + padding)

        return image[top:bottom, left:right]

    def cleanup(self):
        self.face_mesh.close()