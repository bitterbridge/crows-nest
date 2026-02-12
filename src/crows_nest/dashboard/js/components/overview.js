// Overview component

async function refreshOverview() {
    try {
        const data = await api.getOverview();

        document.getElementById('stat-active-thunks').textContent = data.active_thunks;
        document.getElementById('stat-completed-thunks').textContent = data.completed_thunks;
        document.getElementById('stat-failed-thunks').textContent = data.failed_thunks;
        document.getElementById('stat-active-agents').textContent = data.active_agents;
        document.getElementById('stat-queue-depth').textContent = data.queue_depth;
        document.getElementById('stat-uptime').textContent = formatUptime(data.uptime_seconds);
    } catch (e) {
        console.error('Failed to refresh overview:', e);
    }
}

function formatUptime(seconds) {
    if (seconds < 60) {
        return `${Math.floor(seconds)}s`;
    } else if (seconds < 3600) {
        return `${Math.floor(seconds / 60)}m`;
    } else if (seconds < 86400) {
        return `${Math.floor(seconds / 3600)}h`;
    } else {
        return `${Math.floor(seconds / 86400)}d`;
    }
}

// Recent events in overview
const overviewEvents = [];
const maxOverviewEvents = 10;

function addOverviewEvent(event) {
    overviewEvents.unshift(event);
    if (overviewEvents.length > maxOverviewEvents) {
        overviewEvents.pop();
    }
    renderOverviewEvents();
}

function renderOverviewEvents() {
    const container = document.getElementById('overview-events');
    if (!container) return;

    container.innerHTML = overviewEvents.map(event => `
        <div class="event-item">
            <span class="event-type">${event.event_type}</span>
            <span class="event-time">${formatTime(event.timestamp)}</span>
        </div>
    `).join('');
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

// Export functions
window.refreshOverview = refreshOverview;
window.addOverviewEvent = addOverviewEvent;
