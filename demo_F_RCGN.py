import cv2
import face_recognition

# Load images
image_of_person_1 = face_recognition.load_image_file("person_1.jpg")
image_of_person_2 = face_recognition.load_image_file("person_2.jpg")

# Get face encoding for each person
person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]
person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]

# Create a list of known face encodings
known_face_encodings = [
    person_1_face_encoding,
    person_2_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Load test image and find faces in it
test_image = face_recognition.load_image_file("test_image.jpg")
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Loop through each face found in the test image
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # If a match was found in known face(s), use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = "Person " + str(first_match_index + 1)

    face_names.append(name)

# Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(test_image, name, (left, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("Face Recognition", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# we first load the images of the persons whose faces we want to recognize and get their face encodings using the face_encodings function of the face_recognition library. 
# We then create a list of known face encodings.

# Next, we load a test image and find the faces in it using the face_locations function of the face_recognition library. 
# We get the face encodings for each face found using the face_encodings function and compare them with the known face encodings using the compare_faces function. 
# If a match is found, we assign the corresponding name to the face.

# Finally, we display the results by drawing rectangles around the faces and putting the names on them using the cv2.
# rectangle and cv2.putText functions of the OpenCV library.