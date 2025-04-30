function fetchPredictions() {
    fetch('/predict')
        .then(res => res.json())
        .then(data => {
            const tbody = document.querySelector("#predictionTable tbody");
            tbody.innerHTML = '';
            data.forEach(p => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${p.time}</td>
                    <td>${p.interval}</td>
                    <td>${p.min}</td>
                    <td>${p.mid}</td>
                    <td>${p.max}</td>
                    <td>${p.confidence}</td>
                    <td class="feedback-buttons" id="feedback-${p.id}">
                        <button onclick="sendFeedback(${p.id}, true)">👍</button>
                        <button onclick="sendFeedback(${p.id}, false)">👎</button>
                    </td>`;
                tbody.appendChild(row);
            });
        });
}

function sendFeedback(id, isAccurate) {
    fetch('/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ prediction_id: id, is_accurate: isAccurate })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            const buttons = document.querySelector(`#feedback-${id}`);
            if (buttons) {
                buttons.querySelectorAll('button').forEach(btn => {
                    btn.style.opacity = "0.5";
                });
                if (isAccurate) {
                    buttons.querySelector('button:nth-child(1)').style.opacity = "1";
                }
            }
            alert('Feedback updated!');
        } else {
            console.error(data);
            alert('Error submitting feedback.');
        }
    });
}

function checkSystemStatus() {
    fetch('/status')
        .then(res => res.json())
        .then(data => {
            const status = document.getElementById('status');
            const recordCount = document.getElementById('recordCount');

            if (data.online) {
                status.innerHTML = "✅ Online";
            } else {
                status.innerHTML = "❌ Offline";
            }

            recordCount.innerHTML = `Records: ${data.records}`;
        })
        .catch(err => {
            console.error('Error checking status:', err);
            document.getElementById('status').innerHTML = "❌ Offline";
            document.getElementById('recordCount').innerHTML = `Records: 0`;
        });
}

window.onload = function() {
    checkSystemStatus();
};
