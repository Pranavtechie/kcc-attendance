"""
Enroll faces from a JSON file into the face database.

This script reads a JSON file containing person data, downloads the associated
pictures, extracts face embeddings, and adds them to the database.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import requests
from src.engine import FaceEngine
from src.face_db import FaceDatabase
from src.schema import Person
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Constants ---
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "enrolled_faces"
JSON_FILE = DATA_DIR / "attendance-people-data.json"


def enroll_faces():
    """Main function to enroll faces from the JSON file."""
    # Ensure images directory exists
    IMAGES_DIR.mkdir(exist_ok=True)

    # Initialize FaceEngine and FaceDatabase
    try:
        engine = FaceEngine()
        db = FaceDatabase()
        logging.info("Face engine and database initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize face engine or database: {e}")
        return

    # Load person data from JSON
    if not JSON_FILE.exists():
        logging.error(f"JSON file not found: {JSON_FILE}")
        return
    with open(JSON_FILE) as f:
        people_data = json.load(f)

    logging.info(f"Found {len(people_data)} people to enroll.")

    enrollment_count = 0
    for person in tqdm(people_data, desc="Enrolling faces"):
        person_id = person.get("personId")
        if not person_id:
            logging.warning("Skipping entry with no personId.")
            continue

        # Check if person already exists
        if Person.get_or_none(Person.uniqueId == person_id):
            logging.info(f"Person {person_id} already exists. Skipping.")
            continue

        image_url = person.get("picture")
        if not image_url:
            logging.warning(f"No picture URL for person {person_id}. Skipping.")
            continue

        # Download image
        try:
            logging.info(f"Downloading image from {image_url} for person {person_id}")
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_data = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download image for {person_id}: {e}")
            continue
        except Exception as e:
            logging.error(f"Failed to process image for {person_id}: {e}")
            continue

        # Detect and extract face embedding
        faces_feats = engine.detect_and_extract(frame, top_k=1)
        if not faces_feats:
            logging.warning(f"No face detected for person {person_id}. Skipping.")
            continue

        _face, feat = faces_feats[0]

        # Save image locally
        image_filename = f"{person_id}.jpg"
        image_path = IMAGES_DIR / image_filename
        cv2.imwrite(str(image_path), frame)

        # Add person to database
        try:
            db.add_person(
                embedding=feat,
                person_id=person_id,
                name=person["preferredName"],
                person_type=person["userType"],
                admission_number=person.get("admissionNumber"),
                room_id=person.get("roomId"),
                picture_filename=image_filename,
            )
            enrollment_count += 1
            logging.info(f"Successfully enrolled person {person_id}.")
        except Exception as e:
            logging.error(f"Failed to add person {person_id} to database: {e}")
            # Clean up saved image if DB insert fails
            image_path.unlink(missing_ok=True)
            continue

    db.close()
    logging.info("--- Enrollment complete ---")
    logging.info(f"Successfully enrolled {enrollment_count} new people.")


if __name__ == "__main__":
    enroll_faces()
