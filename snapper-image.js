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
            console.log(formData);
            formData.append("image",message.image); // Append the image data to the form

            // Send the form data to your server
            fetch("http://127.0.0.1:5000/submit_report", {
                method: "POST",
                body: formData,
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
