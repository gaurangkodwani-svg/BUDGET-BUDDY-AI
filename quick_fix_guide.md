# Quick Fix Guide: Face Recognition Issues

## Problem Summary
The face recognition system is broken because:
1. `metadata.json` references `gk_000.jpg` 
2. **No actual `.jpg` files exist** in the face_database
3. Empty face database means no training data exists

## Step-by-Step Fix

### Step 1: Clean Up Corrupted Files

Create this script to clean up properly:

```python
#!/usr/bin/env python3
import os
from pathlib import Path

FACE_DB_DIR = Path("face_database")

# Remove corrupted/small recognizer files
corrupted_files = ["recognizer.pkl"]  # Empty file
for file in corrupted_files:
    file_path = FACE_DB_DIR / file
    if file_path.exists():
        size = file_path.stat().st_size
        if size < 10000:  # Less than 10KB (corrupted)
            print(f"Removing corrupted {file} ({size} bytes)")
            file_path.unlink()

print("Cleanup complete")
```

### Step 2: Register a Face User

**Option A: Through Streamlit App (Recommended)**

1. Run the app:
   ```bash
   streamlit run app.py
   ```

2. Click "Register My Face" or "Login with Face"

3. **To Register a Face**:
   - Enter a username (e.g., "admin")
   - Click the camera button
   - Position your face in the frame
   - Click "Save & Register"

4. **The system will automatically**:
   - Save your face image as `admin_000.jpg`
   - Update `metadata.json` with the user
   - Train the face recognizer
   - Create/retrain `label_map.pkl`
   - Create/retrain `recognizer.yml`

**Option B: Direct Face Capture**

If you want to capture a face directly:

```python
import cv2
import numpy as np
from pathlib import Path

# Initialize camera
face_cascade = cv2.CascadeClassifier("face_database/haarcascade_frontalface_default.xml")
if face_cascade is None:
    print("ERROR: Could not load cascade classifier")
    exit(1)

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 'c' to capture face, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display frame
    cv2.imshow('Face Capture - Position your face', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            # Use the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Save the face
            face_dir = Path("face_database")
            face_dir.mkdir(exist_ok=True)
            
            # Count existing images for username
            username = "admin"
            existing_images = list(face_dir.glob(f"{username}_*.jpg"))
            count = len(existing_images)
            filename = f"{username}_{count:03d}.jpg"
            
            cv2.imwrite(str(face_dir / filename), face_roi)
            print(f"✓ Face saved as: {filename}")
            
            # Update metadata
            import json
            metadata_path = face_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"users": {}}
            
            if username not in metadata["users"]:
                metadata["users"][username] = {"face_files": []}
            
            metadata["users"][username]["face_files"].append(filename)
            metadata["users"][username]["registered_at"] = str(pd.Timestamp.now())
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Metadata updated for user: {username}")
            
            # Train recognizer (manual version)
            from app import train_recognizer
            recognizer = train_recognizer()
            if recognizer:
                print("✓ Face recognizer trained successfully!")
            else:
                print("⚠ Training failed - check if face images exist")
        else:
            print("⚠ No face detected. Please try again.")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Step 3: Verify the Fix

```python
#!/usr/bin/env python3
from pathlib import Path

face_db = Path("face_database")

print("Verifying face recognition setup...\n")

# Check required files
required_files = [
    "haarcascade_frontalface_default.xml",
    "metadata.json", 
    "recognizer.yml"
]

for file in required_files:
    path = face_db / file
    if path.exists():
        size = path.stat().st_size
        print(f"✓ {file}: {size:,} bytes")
    else:
        print(f"✗ {file}: MISSING")

# Check face images
face_images = list(face_db.glob("*.jpg"))
print(f"\n✓ Face images found: {len(face_images)}")

# Show registered users
import json
metadata_path = face_db / "metadata.json"
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if metadata.get("users"):
        print(f"\n✓ Registered users:")
        for username, user_data in metadata["users"].items():
            face_count = len(user_data.get("face_files", []))
            print(f"  - {username}: {face_count} face image(s)")
    else:
        print("\n⚠ No registered users in metadata")
```

## Expected Output After Fix

```
Verifying face recognition setup...

✓ haarcascade_frontalface_default.xml: 930,127 bytes
✓ metadata.json: 19 bytes
✓ recognizer.yml: 300,240 bytes

✓ Face images found: 1
✓ Registered users:
  - admin: 1 face image(s)
```

## Troubleshooting

If face recognition still doesn't work:

1. **Check OpenCV installation**:
   ```bash
   pip install opencv-python
   ```

2. **Manual training** (if app registration fails):
   ```python
   from app import train_recognizer
   recognizer = train_recognizer()
   if recognizer:
       print("✓ Recognizer trained successfully")
   else:
       print("✗ Training failed - check if face images exist")
   ```

3. **Test face detection**:
   ```python
   from app import detect_face
   test_image = cv2.imread("test_face.jpg")
   result = detect_face(test_image)
   if result:
       print("✓ Face detection works")
   else:
       print("✗ Face detection failed")
   ```

## Final Verification

After fixing, go to the app and:
1. **Login page** - try face recognition
2. **Register page** - register a new face

Both should work with your registered face!