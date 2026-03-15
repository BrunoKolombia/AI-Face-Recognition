import os
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class FaceDatabase:

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

        self.faces_dir = self.data_dir / "faces"

        self.index_file = self.data_dir / "index.json"

        self.faces_dir.mkdir(parents=True, exist_ok=True)

        self.index = {}

        self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
            self._save_index()

    def _save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, indent=2, fp=f)


    @staticmethod
    def _sanitize_name(name: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in name)

        while "__" in safe:
            safe = safe.replace("__", "_")

        return safe.strip("_").lower()

    def save_face(self, name: str, face_image: np.ndarray, embedding: np.ndarray) -> bool:
        safe_name = self._sanitize_name(name)

        person_dir = self.faces_dir / safe_name

        person_dir.mkdir(exist_ok=True)

        face_path = person_dir / "face.jpg"
        cv2.imwrite(str(face_path), face_image)

        embedding_path = person_dir / "embedding.npy"
        np.save(str(embedding_path), embedding)

        self.index[safe_name] = {
            "display_name": name,
            "face_path": str(face_path),
            "embedding_path": str(embedding_path)
        }

        self._save_index()
        return True

    def get_all_faces(self) -> List[Dict]:
        faces = []

        for safe_name, data in self.index.items():
            faces.append({
                "name": safe_name,
                "display_name": data.get("display_name", safe_name),
                "face_path": data["face_path"],
                "embedding_path": data["embedding_path"]
            })

        return faces

    def load_embedding(self, name: str) -> Optional[np.ndarray]:
        if name in self.index:
            embedding_path = self.index[name]["embedding_path"]

            if os.path.exists(embedding_path):
                return np.load(embedding_path)

        return None

    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings = {}

        for safe_name in self.index.keys():
            embedding = self.load_embedding(safe_name)
            if embedding is not None:
                embeddings[safe_name] = embedding

        return embeddings

    def find_closest_match(self, query_embedding: np.ndarray,
                           threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        embeddings = self.load_all_embeddings()

        if not embeddings:
            return None

        best_match = None
        best_distance = float('inf')

        for name, saved_embedding in embeddings.items():
            distance = np.linalg.norm(query_embedding - saved_embedding)

            if distance < best_distance:
                best_distance = distance
                best_match = name

        if best_distance <= threshold:
            # Get display name
            display_name = self.index[best_match].get("display_name", best_match)
            return (display_name, best_distance)

        return None

    def rename_face(self, old_name: str, new_name: str) -> bool:
        if old_name not in self.index:
            return False

        self.index[old_name]["display_name"] = new_name

        self._save_index()
        return True

    def delete_face(self, name: str) -> bool:
        if name not in self.index:
            return False

        person_dir = self.faces_dir / name
        if person_dir.exists():
            import shutil
            shutil.rmtree(person_dir)

        del self.index[name]

        self._save_index()
        return True

    def search_faces(self, query: str) -> List[Dict]:
        query_lower = query.lower()
        results = []

        for face in self.get_all_faces():
            if query_lower in face["display_name"].lower():
                results.append(face)

        return results
