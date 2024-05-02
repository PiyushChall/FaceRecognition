import cv2
import deepface

# Load the face reference image
reference_image_path = "path/to/your/reference_image.jpg"
reference_image = cv2.imread("FaceReference.jpg.jpg")

# Load the pre-trained face recognition model (optional)
model_name = "VGG-Face"  # You can choose other models like "FaceNet", "ArcFace", etc.
model = deepface.load_model(model_name)

# Define the face detection cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Capture video from webcam (or provide a video path)
cap = cv2.VideoCapture(0)  # 0 for webcam, replace with video path if using a video

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face_region = frame[y:y + h, x:x + w]

        # Resize the face region for model compatibility
        resized_face = cv2.resize(face_region, (152, 152))

        # Perform face recognition
        result = deepface.verify(
            img1_path=reference_image_path,
            img2_path=resized_face,
            model_name=model_name,
            detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        )

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if result["verified"]:
            cv2.putText(
                frame, f"Recognized: {result['identity']}", (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                frame, f"Unknown", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2
            )

    # Display the frame with bounding boxes and labels
    cv2.imshow("Face Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
