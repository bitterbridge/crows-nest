// Event stream component

const allEvents = [];
const maxEvents = 100;
let eventFilter = '';

function addEvent(event) {
    allEvents.unshift(event);
    if (allEvents.length > maxEvents) {
        allEvents.pop();
    }
    renderEvents();

    // Also add to overview
    if (window.addOverviewEvent) {
        window.addOverviewEvent(event);
    }
}

function filterEvents() {
    eventFilter = document.getElementById('event-filter')?.value?.toLowerCase() || '';
    renderEvents();
}

function clearEvents() {
    allEvents.length = 0;
    renderEvents();
}

function renderEvents() {
    const container = document.getElementById('event-stream');
    if (!container) return;

    const filteredEvents = eventFilter
        ? allEvents.filter(e =>
            e.event_type.toLowerCase().includes(eventFilter) ||
            JSON.stringify(e.payload).toLowerCase().includes(eventFilter)
        )
        : allEvents;

    if (filteredEvents.length === 0) {
        container.innerHTML = '<p class="placeholder">No events yet. Events will appear here in real-time.</p>';
        return;
    }

    container.innerHTML = filteredEvents.map(event => `
        <div class="event-item">
            <div>
                <span class="event-type">${event.event_type}</span>
                <span class="event-time">${formatEventTime(event.timestamp)}</span>
            </div>
            <div class="event-payload">${formatPayload(event.payload)}</div>
        </div>
    `).join('');
}

function formatEventTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString() + '.' + date.getMilliseconds().toString().padStart(3, '0');
}

function formatPayload(payload) {
    if (!payload || Object.keys(payload).length === 0) {
        return '';
    }
    // Show abbreviated payload
    const str = JSON.stringify(payload);
    return str.length > 100 ? str.substring(0, 100) + '...' : str;
}

// Export functions
window.addEvent = addEvent;
window.filterEvents = filterEvents;
window.clearEvents = clearEvents;
