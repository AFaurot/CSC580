import os
import face_recognition
import PIL.Image
import PIL.ImageDraw


# Function to check if a known person is in a crowd image
def is_person_in_crowd(known_face_path, crowd_image_path, tolerance: float = 0.6, save_image=False, show_image=True):
    # Load and encode the known face
    known_image = face_recognition.load_image_file(known_face_path)
    known_encodings = face_recognition.face_encodings(known_image)

    if len(known_encodings) == 0:
        raise ValueError("No face found in the known face image. Use a clearer photo with one face.")
    if len(known_encodings) > 1:
        print("Warning: More than one face found in the known image. Using the first face.")

    known_encoding = known_encodings[0]

    # Load crowd image and find faces
    crowd_image = face_recognition.load_image_file(crowd_image_path)
    crowd_locations = face_recognition.face_locations(crowd_image)
    crowd_encodings = face_recognition.face_encodings(crowd_image, crowd_locations)

    print(f"Found {len(crowd_locations)} face(s) in the crowd image.")

    # Compare
    match_found = False
    matching_locations = []

    for (loc, enc) in zip(crowd_locations, crowd_encodings):
        matches = face_recognition.compare_faces([known_encoding], enc, tolerance=tolerance)
        if matches[0]:
            match_found = True
            matching_locations.append(loc)

    # Print result
    if match_found:
        print("Match found: The person was detected in {}".format(crowd_image_path))
    else:
        print("No match: The person was not in {}".format(crowd_image_path))

    # Draw boxes aroung faces (red for matches, blue for others)
    pil_img = PIL.Image.fromarray(crowd_image)
    draw = PIL.ImageDraw.Draw(pil_img)

    for (top, right, bottom, left) in crowd_locations:
        color = "red" if (top, right, bottom, left) in matching_locations else "blue"
        draw.rectangle([left, top, right, bottom], outline=color, width=4)

    if show_image:
        pil_img.show()
    if save_image:
        output_dir = "output_images"
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # Build output filename
        base_name = os.path.basename(crowd_image_path)
        name, ext = os.path.splitext(base_name)
        # Save labeled image
        save_path = os.path.join(output_dir, name + "_detection.jpg")
        pil_img.save(save_path)
        print(f"Labeled image saved to {save_path}")

    return match_found


# Main function loops over images in a folder, checking for the known person
def main():
    # Load image of known person
    known_face_path = "me.jpg" # Path to the known person's image
    crowd_folder = "crowd_images" # Folder containing crowd images
    save_choice = input("Would you like to save the labeled images? (y/n): ")
    save_image_choice = True if save_choice.lower() == 'y' else False
    show_choice= input("Would you like to show the images after processing? (y/n): ")
    show_image_choice = True if show_choice.lower() == 'y' else False
    # Loop over the crowd images in the specified folder
    for filename in os.listdir(crowd_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            crowd_image_path = os.path.join(crowd_folder, filename)
            print(f"Checking crowd image: {filename}")
            is_person_in_crowd(known_face_path,
                               crowd_image_path,
                               tolerance=0.6,
                               save_image=save_image_choice,
                               show_image=show_image_choice)
            print("-" * 60)
            if filename == os.listdir(crowd_folder)[-1]:
                print("No more images to check.")
            else:
                choice = input("Would you like to check the next image? (y/n): ")
                if choice.lower() != 'y':
                    break


if __name__ == "__main__":
    main()
