document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const emailContent = document.getElementById('email-content');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const loader = document.getElementById('loader');
    const resultText = document.getElementById('result-text');

    form.addEventListener('submit', async (event) => {
        // Prevent the default form submission which reloads the page
        event.preventDefault();

        const emailText = emailContent.value;
        if (!emailText.trim()) {
            alert('Please paste some email content to analyze.');
            return;
        }

        // Show the result container and loader, disable the button
        resultContainer.classList.remove('hidden');
        loader.classList.remove('hidden');
        resultText.innerHTML = ''; // Clear previous results
        submitBtn.disabled = true;
        submitBtn.textContent = 'Analyzing...';

        try {
            // Send the text content to the backend API using fetch
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email_text: emailText }),
            });

            const data = await response.json();

            // Display the result
            displayResult(data);

        } catch (error) {
            // Handle network errors
            displayResult({ error: 'An error occurred. Please try again.' });
            console.error('Error:', error);
        } finally {
            // Re-enable the button and hide the loader
            loader.classList.add('hidden');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Analyze Email';
        }
    });

    function displayResult(data) {
        if (data.error) {
            resultText.innerHTML = `<p class="result-error">Error: ${data.error}</p>`;
        } else {
            let predictionClass = 'result-inconclusive';
            if (data.prediction === 'Phishing') {
                predictionClass = 'result-phishing';
            } else if (data.prediction === 'Legitimate') {
                predictionClass = 'result-legitimate';
            }

            resultText.innerHTML = `
                <p class="${predictionClass}">Prediction: ${data.prediction}</p>
                <p>Confidence: ${data.confidence}</p>
            `;
        }
    }
});