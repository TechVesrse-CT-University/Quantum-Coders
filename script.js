document.getElementById("file-upload-form").addEventListener("submit", function (e) {
    e.preventDefault();

    const fileInput = document.getElementById("file");
    const file = fileInput.files[0];
    const errorMessage = document.getElementById("error-message");
    const loadingMessage = document.getElementById("loading");
    const predictionResult = document.getElementById("prediction-result");
    const predictionList = document.getElementById("prediction-list");

    // Clear previous results
    predictionList.innerHTML = '';
    errorMessage.innerHTML = '';
    loadingMessage.style.display = 'block';

    if (!file) {
        errorMessage.textContent = "Please select a file.";
        loadingMessage.style.display = 'none';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Replace this URL with your ngrok URL or your API endpoint
    const apiUrl = "http://<ngrok_subdomain>.ngrok.io/predict";

    fetch(apiUrl, {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        loadingMessage.style.display = 'none';

        if (data.error) {
            errorMessage.textContent = data.error;
        } else if (data.prediction && data.prediction.length > 0) {
            data.prediction.forEach(question => {
                const listItem = document.createElement('li');
                listItem.textContent = question;
                predictionList.appendChild(listItem);
            });
            predictionResult.style.display = 'block';
        } else {
            errorMessage.textContent = "No predictions available.";
        }
    })
    .catch(error => {
        loadingMessage.style.display = 'none';
        errorMessage.textContent = "An error occurred while processing the file.";
    });
});
