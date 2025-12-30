// Add dark mode
const darkModeToggle = document.getElementById('darkMode');
darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
});

// Add progressive web app (PWA)
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js');
}

// Real-time validation
document.getElementById('doctorReport').addEventListener('input', (e) => {
    if (e.target.value.length < 20) {
        e.target.style.borderColor = 'orange';
    } else {
        e.target.style.borderColor = 'green';
    }
});

// Download results as CSV
function downloadCSV(data) {
    const csv = `Prediction,Confidence,Similarity,Warning\n${data.ai_prediction},${data.confidence_percent},${data.similarity_score},${data.warning_flag}`;
    const a = document.createElement('a');
    a.href = 'data:text/csv,' + encodeURIComponent(csv);
    a.download = 'results.csv';
    a.click();
}