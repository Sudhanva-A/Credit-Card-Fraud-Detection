document.getElementById('predict-btn').addEventListener('click', async () => {
    const features = Array.from(document.querySelectorAll('.feature-input'))
        .map(input => parseFloat(input.value));
    
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
    });
    
    const result = await response.json();
    document.getElementById('result').innerHTML = `
        Probability: ${(result.probability * 100).toFixed(2)}%
        ${result.is_fraud ? '🚨 FRAUD' : '✅ LEGIT'}
    `;
});