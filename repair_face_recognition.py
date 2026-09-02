#!/usr/bin/env python3
"""
Face Recognition Repair Script for BudgetBuddy AI
This script fixes issues with face recognition by:
1. Fixing corrupted metadata.json
2. Removing invalid label_map.pkl if corrupted
3. Providing guidance for retraining
"""

import json
import pickle
import os
from pathlib import Path

# Configuration
FACE_DB_DIR = Path("face_database")
METADATA_FILE = FACE_DB_DIR / "metadata.json"
LABEL_MAP_FILE = FACE_DB_DIR / "label_map.pkl"
CASCADE_FILE = FACE_DB_DIR / "haarcascade_frontalface_default.xml"
RECOGNIZER_FILE = FACE_DB_DIR / "recognizer.yml"

def check_cascade_file():
    """Verify haarcascade file exists and is valid."""
    if CASCADE_FILE.exists():
        size = CASCADE_FILE.stat().st_size
<<<<<<< HEAD
        print(f"✓ Cascade file exists: {CASCADE_FILE.name} ({size:,} bytes)")
        return True
    else:
        print(f"✗ Cascade file missing: {CASCADE_FILE}")
=======
        print("[+] Cascade file exists: {} ({} bytes)".format(CASCADE_FILE.name, size))
        return True
    else:
        print("[x] Cascade file missing: {}".format(CASCADE_FILE))
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)
        return False

def fix_metadata():
    """Fix metadata.json by removing references to missing face files."""
    if not METADATA_FILE.exists():
<<<<<<< HEAD
        print("✗ metadata.json not found")
=======
        print("[x] metadata.json not found")
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)
        return

    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    original_users = list(metadata.get("users", {}).keys())
    users_to_remove = []

    for username, user_data in metadata.get("users", {}).items():
        for face_file in user_data.get("face_files", []):
            face_path = FACE_DB_DIR / face_file
            if not face_path.exists():
<<<<<<< HEAD
                print(f"  ⚠ Missing face file: {face_file} (user: {username})")
=======
                print("  [!] Missing face file: {} (user: {})".format(face_file, username))
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)
                users_to_remove.append(username)
                break

    # Fix metadata
    for username in set(users_to_remove):
        del metadata["users"][username]
<<<<<<< HEAD
        print(f"  ✓ Removed user '{username}' from metadata (missing face files)")
=======
        print("  [+] Removed user '{}' from metadata (missing face files)".format(username))
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)

    if users_to_remove:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
<<<<<<< HEAD
        print(f"\n✓ metadata.json fixed")
    else:
        print(f"\n✓ All {len(original_users)} users have valid face files")
=======
        print("\n[+] metadata.json fixed")
    else:
        print("\n[+] All {} users have valid face files".format(len(original_users)))
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)

def check_label_map():
    """Check if label_map.pkl is valid."""
    if not LABEL_MAP_FILE.exists():
<<<<<<< HEAD
        print("✓ label_map.pkl does not exist (will be regenerated on re-registration)")
=======
        print("[+] label_map.pkl does not exist (will be regenerated on re-registration)")
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)
        return True

    try:
        with open(LABEL_MAP_FILE, 'rb') as f:
            data = pickle.load(f)
<<<<<<< HEAD
        print(f"✓ label_map.pkl is valid: {data}")
        return True
    except Exception as e:
        print(f"✗ label_map.pkl is corrupted: {e}")
        print("  This file should be deleted and regenerated automatically")
=======
        print("[+] label_map.pkl is valid: {}".format(LABEL_MAP_FILE))
        return True
    except Exception as e:
        print("[!] label_map.pkl is corrupted: {}".format(e))
        print("    This file should be deleted and regenerated automatically")
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)
        return False

def check_recognizer():
    """Check if recognizer model exists and is valid."""
    if not RECOGNIZER_FILE.exists():
<<<<<<< HEAD
        print("✓ recognizer.yml does not exist (will be regenerated on registration)")
        return True

    size = RECOGNIZER_FILE.stat().st_size
    if size < 1000:
        print(f"✗ recognizer.yml seems corrupted (only {size} bytes, expected ~300KB+)")
        return False

    print(f"✓ recognizer.yml exists ({size:,} bytes)")
    return True

def get_face_image_count():
    """Count valid face images in database."""
    face_images = list(FACE_DB_DIR.glob("*.jpg"))
    print(f"\n📊 Face images in database: {len(face_images)}")
    for img in face_images:
        print(f"  - {img.name}")
    return len(face_images)

def main():
    print("=" * 60)
    print("🔍 Face Recognition Repair Tool")
    print("=" * 60)

    print(f"\n📁 Database directory: {FACE_DB_DIR.absolute()}")

    # Check cascade
    print("\n1. Checking cascade file...")
    cascade_ok = check_cascade_file()

    # Fix metadata
    print("\n2. Checking metadata.json...")
    fix_metadata()

    # Check label map
    print("\n3. Checking label_map.pkl...")
    label_ok = check_label_map()

    # Check recognizer
    print("\n4. Checking recognizer model...")
    recog_ok = check_recognizer()

    # Count face images
    print("\n5. Checking face images...")
    image_count = get_face_image_count()

    # Summary
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
=======
        print("[+] recognizer.yml does not exist (will be regenerated on registration)")
        return True

    size = RECOGNIZER_FILE.stat().st_size
    if size < 1000:  # Less than 1KB (expected is ~300KB+)
        print("[!] recognizer.yml seems corrupted (only {} bytes, expected ~300KB+)".format(size))
        return False

    print("[+] recognizer.yml exists ({} bytes)".format(size))
    return True

def count_face_images():
    """Count valid face images in database."""
    face_images = list(FACE_DB_DIR.glob("*.jpg"))
    print("\n[*] Face images in database: {}".format(len(face_images)))
    for img in face_images:
        print("  - {}".format(img.name))
    return len(face_images)

def main():
    print("\n" + "[" + "="*60 + "]")
    print("[" + "Face Recognition Repair Tool" + " " + "]" + " ")
    print("[" + "="*60 + "]" + "\n")

    print("[i] Database directory: {}".format(FACE_DB_DIR.absolute()))

    # Check cascade
    print("\n[1] Checking cascade file...")
    cascade_ok = check_cascade_file()

    # Fix metadata
    print("\n[2] Checking metadata.json...")
    fix_metadata()

    # Check label map
    print("\n[3] Checking label_map.pkl...")
    label_ok = check_label_map()

    # Check recognizer
    print("\n[4] Checking recognizer model...")
    recog_ok = check_recognizer()

    # Count face images
    print("\n[5] Checking face images...")
    image_count = count_face_images()

    # Summary
    print("\n" + "[" + "-"*60 + "]")
    print("[" + "Summary" + " " + "]" + " ")
    print("[" + "-"*60 + "]" + "\n")
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)

    issues = []
    if not cascade_ok:
        issues.append("Cascade file missing/invalid")
    if not label_ok:
        issues.append("Label map corrupted - needs regeneration")
    if not recog_ok:
        issues.append("Recognizer model corrupted - needs regeneration")
    if image_count == 0:
        issues.append("No face images registered")

    if issues:
<<<<<<< HEAD
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")

        print("\n✅ Recommended fix:")
=======
        print("[!] Issues found:")
        for issue in issues:
            print("  - {}".format(issue))

        print("\n[+] Recommended fix:")
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)
        print("  1. Go to the app's 'Register' page")
        print("  2. Enter your username and capture a new face")
        print("  3. Click 'Save & Register'")
        print("\n  The system will automatically retrain the recognizer!")
    else:
<<<<<<< HEAD
        print("✅ All face recognition components are healthy!")

    print("=" * 60)
=======
        print("[+] All face recognition components are healthy!")

    print("\n" + "[" + "="*60 + "]\n")
>>>>>>> a2ad767 (Initial commit: BudgetBuddy AI personal finance dashboard with face auth and ML analytics)

if __name__ == "__main__":
    main()