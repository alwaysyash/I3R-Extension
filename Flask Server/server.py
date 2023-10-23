from flask import Flask, request, jsonify,render_template
import json
import base64

import firebase_admin 
from firebase_admin import credentials
from firebase_admin import firestore
import datetime

app = Flask(__name__)

@app.route('/submit_report', methods=['GET','POST'])
def submit_report():
    # Receive the form data and image from the Chrome extension
    form_data = request.form #can be used for database
    #for local operations
    report_tags=[]
    platforms=[]
    user=[]

    # Separate the form data from the image
    for key, value in form_data.items():
        if key == 'image':
            # Handle the image separately
            image = value
        elif key.startswith('report_reason['):
            report_tags.append(value)
        elif key=='report_platform':
            platforms.append(value)
        elif key=='custom_platform_value' and value!='':
            platforms.append(value)
        elif key =='user':
            user.append(value)
    
    # Process the form data as needed
    print("Form Data:")
    print(json.dumps(form_data, indent=4))
    print("Tags:",report_tags)
    print("Platform:",platforms)
    print("User:",user)

    # Image decode and save
    base64_image_data = image
    base64_image_data = base64_image_data.split(',')[1] # Remove the 'data:image/png;base64,' prefix
    image_data = base64.b64decode(base64_image_data)
    file_path = 'Flask Server\Images\image.png'
    with open(file_path, 'wb') as image_file:
        image_file.write(image_data)

    print("Image saved successfully to", file_path)


#Firebase Integration
    key={
        "type": "service_account",
        "project_id": "i3r-extension",
        "private_key_id": "462c8f6c7bc58880508571756c4408be6aa08658",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC3jZkgXpf+Q1wk\nlWRHdTwyRWuJHSgwSRxygMN8ZRdMHaOz+CVnXXOrywmFDVngQBONWqD3gxEwY62O\nSWzzcGtsIbdTlX9hVXlyIi9cERFluwxBK5cQKHgXSeTO8uTBHkDslvVaDwrkrNSs\nL+R0PhQQcBEcIC98KM6Y5Jh9FHZSw4iXsFuFOqd6hzN+yUdtf5QOjktjQFrewAhR\nu62sXyUKwoWth89zYlvnICQ89ElxGC8nXzzjonFX41Q8mL/mATtrdsaPGd9B64fj\nKSX79bOEKbQFZNvBAFsOezTwLtqjsflGlhtK/AfviNqaPG0Synhe6CrV9W3sT5c/\nApe+cgFBAgMBAAECggEAAZb0GcvqmJI4RifWEGld7gMxu88DwKv4F1ob906VKFIE\nZNlevORaDVrq4vBbPtOHPSeTsnkRM8r/3zYtv6D8XoQfPJXUSSvulelhDB9+DVRp\n9iwyDiWE6WdjgtL8iLCh+CU+4ZAKSdyLqKRvNSjZec1NVKHxugCiI2sCvWiT4BfG\nKVa3tkF19AkDvRx96cmOHCieHlVi1sAP1b/ZiXyBoRLxiIfbMFPRqXnq26Wzy7kv\n8UtduvoK4yV5cLQtKKcVEmtMxLwk3NVmrJPxv25JH92Zm+3BS/2D5J0OAOgkzBvh\nuBTT2T8C9XlidLT56CzVIuEQcnCSIm3FlWTvIfh+zQKBgQDrY0zn92I2iwJLZJWg\nRTHqpCFzbvjdbgtmu3XFtMcli+61BDgXVO4NUZ5BUqjao+M7RVoRA/T0q2/lADTS\ndoIn7fwXRakYxBMUoywpJBnc/hEva7u5xEYrWZYLWq/IgctnSdL1eLwXNvNDDzl/\n+8jFXxa3ZS/hZoyKcFyIUK/YFQKBgQDHoFDJJmfGq/QdFGwYsT6Pkp1uK8EndOKF\n7vfvju8bwvnhD4VLHHVwd2cMpPVZL3cHlhtpSQ8o6+I8wJqQj9MXxjqXN0nKv7pg\nsRtl/Z4r7T50MxyImYv4ZZL70j6X8DTAiSndbzxG80oc25bjs0HoDP7lXNfISOoY\nuOp0gOhDfQKBgA/jvCVMhcy4xYbhW1heU2hLfBaWvyXzb8GXlfOqqGbYc1y01DR+\n/zLW95/hPJTFy2kM3W+YnMiHah5DTU2HwnF/lyza/vc1BTS3bxu33CcW34Ib+6gm\nn7X9biuGC6e1W021pg/7/nZytykntfH1xS/No3Lt2bWVfpc5zoVFLRmhAoGAJ+4c\nWq/w8B9zG+H64VGK1wMXtHLSFwddTDcJpwRrNZ1hiDAnlGej4hQwK6pPXUCOSZkv\n8HZpruDIByjrgeES8112WMr5WrHRIQgsF9GMMvMom+uhWH2GLvB6Xx6l4JRuqNiG\n5EEcyIBfobgWzYMMutLmpZBVpT1Yfgt28kjarK0CgYEAnGv1rVF8iUzIxUE/u2O6\nW1e6aUci/kYkmkRm3xqfQW1ZZ+G9rr5ogMBryYyX7PReqXqaqQ8pFlIt0eLwNamO\nd6vwil8XODorq4p/2kUCBuUZkuFOIZtv0nXKSOyDKjd2KkAmppEAmdnn9npJnch7\nxBZWWWUt1w0pK4iLYURuEdg=\n-----END PRIVATE KEY-----\n",
        "client_email": "firebase-adminsdk-11h28@i3r-extension.iam.gserviceaccount.com",
        "client_id": "114711825863426214473",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-11h28%40i3r-extension.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }
    cred=credentials.Certificate(key)
    default_app = firebase_admin.initialize_app(cred)
    print(default_app.name)

    db=firestore.client()
    timestamp=datetime.datetime.now()
    data={
        "DateTime": timestamp,
        "Tags":report_tags,
        "Platform": platforms[0],
        "User": user[0]
    }

    doc_ref = db.collection("I3R")
    doc_ref.add(data)

   # Respond with a JSON message (you can customize this response)
    response = {"message": "Report received and processed"}
    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True)





#     # Process the form data and image as needed
#     # You can access form fields like form_data['report_reason[]']
#     # You can access the image using image.read() or save it to a file





