// Function to predict message spam
async function predictMessage() {
    const message = document.getElementById('message').value;
    const resultElement = document.getElementById('message-result');

    if (!message) {
        resultElement.textContent = 'Please enter a message.';
        return;
    }

    try {
        const response = await fetch('http://127.0.0.1:8000/predict-message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        resultElement.textContent = `Prediction: ${data.prediction === 1 ? 'Spam' : 'Not Spam'}`;
    } catch (error) {
        console.error('Error:', error);
        resultElement.textContent = 'An error occurred.';
    }
}

// Function to predict URL spam
async function predictURL() {
    const url = document.getElementById('url').value;
    const resultElement = document.getElementById('url-result');

    if (!url) {
        resultElement.textContent = 'Please enter a URL.';
        return;
    }

    try {
        const response = await fetch('http://127.0.0.1:8000/predict-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        resultElement.textContent = `Prediction: ${data.prediction === 1 ? 'Spam' : 'Not Spam'}`;
    } catch (error) {
        console.error('Error:', error);
        resultElement.textContent = 'An error occurred.';
    }
}