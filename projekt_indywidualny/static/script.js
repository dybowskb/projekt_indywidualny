document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();

    let formData = new FormData(this);
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Gatunek: ${data.prediction}`;
    })
    .catch(error => console.error('Error:', error));
});
