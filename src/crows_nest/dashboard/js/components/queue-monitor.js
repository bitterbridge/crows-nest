// Queue monitor component

let currentQueues = [];

async function refreshQueues() {
    try {
        currentQueues = await api.getQueues();
        renderQueues();
    } catch (e) {
        console.error('Failed to refresh queues:', e);
    }
}

function renderQueues() {
    const container = document.getElementById('queue-list');
    if (!container) return;

    if (currentQueues.length === 0) {
        container.innerHTML = '<p class="placeholder">No queues found</p>';
        return;
    }

    container.innerHTML = currentQueues.map(queue => `
        <div class="queue-card">
            <div class="queue-name">${queue.name}</div>
            <div class="queue-stats">
                <div>
                    <div class="queue-stat-value pending">${queue.pending}</div>
                    <div class="queue-stat-label">Pending</div>
                </div>
                <div>
                    <div class="queue-stat-value processing">${queue.processing}</div>
                    <div class="queue-stat-label">Processing</div>
                </div>
                <div>
                    <div class="queue-stat-value dead">${queue.dead}</div>
                    <div class="queue-stat-label">Dead</div>
                </div>
            </div>
        </div>
    `).join('');
}

// Export functions
window.refreshQueues = refreshQueues;
