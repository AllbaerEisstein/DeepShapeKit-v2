import cv2
import numpy as np
import json
import argparse
import os

points = []
max_display_size = 1024
display_scale = 1.0  # To be computed later

def main():
    global img_copy, display_scale
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="input file")
    args = parser.parse_args()
    
    image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.image)

    def redraw_image():
        """Redraws the image with all currently stored points."""
        global img_copy
        img_copy = img_resized.copy()
        i = 0
        for idx, (x, y) in enumerate(points):
            x_scaled = int(x * display_scale)
            y_scaled = int(y * display_scale)
            cv2.circle(img_copy, (x_scaled, y_scaled), 1, (0, 0, 255), -1)
            if i == len(points)-1:
                cv2.putText(img_copy, str(idx + 1), (x_scaled + 5, y_scaled - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
            i = i + 1
        cv2.imshow("Image", img_copy)

    def click_event(event, x, y, flags, param):
        global points, img_copy
        if event == cv2.EVENT_LBUTTONDOWN:  # Click to label points
            x_orig = int(x / display_scale)
            y_orig = int(y / display_scale)
            points.append((x_orig, y_orig))
            print(f"Point {len(points)}: ({x_orig}, {y_orig})")
            redraw_image()

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found!")
        exit()

    h, w = img.shape[:2]
    display_scale = min(max_display_size / max(h, w), 1.0)
    img_resized = cv2.resize(img, (int(w * display_scale), int(h * display_scale)))
    img_copy = img_resized.copy()

    cv2.imshow("Image", img_copy)
    cv2.setMouseCallback("Image", click_event)

    print("Click on the checkerboard corners in order. Press 's' to save, 'r' to reset, 'u' to undo last point, or 'q' to quit.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):  # Save points to JSON
            fname = os.path.splitext(os.path.basename(image_path))[0]
            fdir = os.path.dirname(image_path)
            with open(fdir + "/annotated/" + fname + "_labeled_points.json", "w") as f:
                json.dump({fname: points}, f)
            print(f"Points saved to {fdir}/annotated/{fname}_labeled_points.json")
        elif key == ord('r'):  # Reset all points
            points.clear()
            redraw_image()
            print("Reset all points.")
        elif key == ord('u'):  # Undo last point
            if points:
                removed_point = points.pop()
                print(f"Removed last point: {removed_point}")
                redraw_image()
            else:
                print("No points to undo.")
        elif key == ord('q'):  # Quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

