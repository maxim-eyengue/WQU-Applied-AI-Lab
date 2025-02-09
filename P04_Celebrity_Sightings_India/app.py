# **AI Lab: Deep Learning for Computer Vision**
# **WorldQuant University**
#
#

# **Usage Guidelines**
#
# This file is licensed under Creative Commons Attribution-NonCommercial-
# NoDerivatives 4.0 International.
#
# You **can** :
#
#   * ✓ Download this file
#   * ✓ Post this file in public repositories
#
# You **must always** :
#
#   * ✓ Give credit to WorldQuant University for the creation of this file
#   * ✓ Provide a link to the license
#
# You **cannot** :
#
#   * ✗ Create derivatives or adaptations of this file
#   * ✗ Use this file for commercial purposes
#
# Failure to follow these guidelines is a violation of your terms of service and
# could lead to your expulsion from WorldQuant University and the revocation
# your certificate.
#
#

# Necessary import
import io
from flask import Flask, render_template, request, send_file
from PIL import Image
# Import our face recognition code
from face_recognition import add_labels_to_image

# Starts Flask
app = Flask(__name__)


# Set the route to "/"
@app.route("/")
def home(): # Function for displaying the home page
    return render_template("upload.html") # to set-up the HTML home file putting to be filled slots inside


@app.route("/recognize", methods = ["POST"])
def process_image(): # Function for processing the image
    # Display an error message if no image found
    if "image" not in request.files:
        return "No image provided", 400

    # Get the file sent along with the request
    file = request.files["image"]

    # Video also shows up as an image
    # we want to reject those as well
    if not file.mimetype.startswith("image/"):
        return "Image format not recognized", 400

    # Get the raw bits out of the image data from the uploaded file
    image_data = file.stream

    # Run our face recognition code!
    img_out = run_face_recognition(image_data)

    # Return an error if the image was not processed correctly
    if img_out == Ellipsis:
        return "Image processing not enabled", 200
    else:
        # Our function returns something from matplotlib,
        # convert it to a web-friendly form and return it
        out_stream = matplotlib_to_bytes(img_out)
        return send_file(out_stream, mimetype = "image/jpeg") # to return the processed image


def run_face_recognition(image_data): # Face recognition function
    # Open image_data with PIL
    input_image = Image.open(image_data)

    # Run our function on the PIL image
    img_out = add_labels_to_image(input_image)

    return img_out # to return the labeled image


def matplotlib_to_bytes(figure): # Plotting function
    # Create an in-memory bytes buffer to store the image data
    buffer = io.BytesIO()
    # Save the matplotlib figure to the buffer
    figure.savefig(buffer,
                   format = "jpg", # specifying the output format (JPEG)
                   bbox_inches = "tight") # removing redundant whitespace around the plot
    # Reset buffer position to start (byte 0) after writing
    buffer.seek(0) # allowing subsequent reads to start from the beginning of the data
    return buffer # to return the buffer containing the JPEG image bytes


# If the script is executed
if __name__ == "__main__":
    # Run the Flask app
    app.run(debug = True)
    

# This file © 2024 by WorldQuant University is licensed under CC BY-NC-ND 4.0.