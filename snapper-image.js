"use strict";

// Add listener that recieves the image with additional information
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "sendImageToNewTab") {
        var image = document.getElementById("capture_image");
        // var downloadButtonPNG = document.getElementById("download_png")

        // Set image source as base64 image
        image.src = message.image;

        // Add zoom in/out when image clicked
        image.onclick = () => {
            console.log("image clicked");
            image.classList.contains("zoomed_in")
                ? image.classList.remove("zoomed_in")
                : image.classList.add("zoomed_in");
        };

        Report.addEventListener("click", function () {
            // Capture the form data
            var formData = new FormData(report_form);
            var platformData= new FormData(platform_form);
            var userData=new FormData(user_form);
            
            //A new FormData object to combine the data
var combinedFormData = new FormData();

// Append the entries from each of the individual FormData objects
for (var pair of formData.entries()) {
    combinedFormData.append(pair[0], pair[1]);
}

for (var pair of platformData.entries()) {
    combinedFormData.append(pair[0], pair[1]);
}

for (var pair of userData.entries()) {
    combinedFormData.append(pair[0], pair[1]);
}

            combinedFormData.append("image",message.image); // Append the image data to the form

            // Send the form data to your server
            fetch("http://127.0.0.1:5000/submit_report", {
                method: "POST",
                body: combinedFormData,
            })
                .then((response) => response.json())
                .then((data) => {
                    // Handle the response from the server if needed
                    console.log(data);
                });
        });

        sendResponse(JSON.stringify(message, null, 4) || true);

        return true;
    }
});

// Set href and download property on button to download image when clicked
// downloadButtonPNG.href = message.image
// downloadButtonPNG.download = message.filename
