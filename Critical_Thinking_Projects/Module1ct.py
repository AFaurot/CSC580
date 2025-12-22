import face_recognition
import PIL.Image
import PIL.ImageDraw

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("punta_cana.jpg")

# Find all the faces in the image
faceLocations = face_recognition.face_locations(image)

numberOfFaces = len(faceLocations)
print("Found {} face(s) in this picture.".format(numberOfFaces))

# Load the image into a Python Image Library object so that you can draw on top of it
pilImage = PIL.Image.fromarray(image)
drawHandle = PIL.ImageDraw.Draw(pilImage)

for faceLocation in faceLocations:
    # Each faceLocation is (top, right, bottom, left)
    top, right, bottom, left = faceLocation

    # Print the location of each face
    print(
        "A face is located at pixel location "
        "Top: {}, Left: {}, Bottom: {}, Right: {}".format(
            top, left, bottom, right
        )
    )

    # Draw a red box around the face - I set width to 6 for a thicker border
    drawHandle.rectangle(
        [left, top, right, bottom],
        outline="red",
        width=6
    )

# Display the image on screen
pilImage.show()

# Save the image to a file
pilImage.save("punta_cana_faces.jpg")

