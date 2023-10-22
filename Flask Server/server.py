from flask import Flask, request, jsonify,render_template
import json
import base64

app = Flask(__name__)

@app.route('/submit_report', methods=['GET','POST'])
def submit_report():
    # Receive the form data and image from the Chrome extension
    form_data = request.form
    form_data_dict = {}

    # Separate the form data from the image
    for key, value in form_data.items():
        if key == 'image':
            # Handle the image separately
            image = value
        else:
            # Store other form fields in the form_data_dict
            form_data_dict[key] = value
    
    # Process the form data as needed
    print("Form Data:")
    print(json.dumps(form_data_dict, indent=4))

    base64_image_data = image
    base64_image_data = base64_image_data.split(',')[1] # Remove the 'data:image/png;base64,' prefix
    image_data = base64.b64decode(base64_image_data)
    file_path = 'Flask Server\Images\image.png'
    with open(file_path, 'wb') as image_file:
        image_file.write(image_data)

    print("Image saved successfully to", file_path)


   # Respond with a JSON message (you can customize this response)
    response = {"message": "Report received and processed"}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)





#     # Process the form data and image as needed
#     # You can access form fields like form_data['report_reason[]']
#     # You can access the image using image.read() or save it to a file





