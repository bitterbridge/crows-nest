// Main application entry point

// Current view state
let currentView = 'overview';

// Initialize the application
function init() {
    // Set up navigation
    setupNavigation();

    // Connect to event socket
    if (window.eventSocket) {
        window.eventSocket.connect();
        window.eventSocket.subscribe(handleEvent);
    }

    // Load initial view
    switchView('overview');
}

function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const view = item.dataset.view;
            if (view) {
                switchView(view);
            }
        });
    });
}

function switchView(viewName) {
    currentView = viewName;

    // Update nav highlighting
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });

    // Hide all views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
    });

    // Show selected view
    const targetView = document.getElementById(`${viewName}-view`);
    if (targetView) {
        targetView.classList.add('active');
    }

    // Load view data
    loadViewData(viewName);
}

function loadViewData(viewName) {
    switch (viewName) {
        case 'overview':
            if (window.refreshOverview) window.refreshOverview();
            break;
        case 'thunks':
            if (window.refreshThunks) window.refreshThunks();
            break;
        case 'agents':
            if (window.refreshAgents) window.refreshAgents();
            break;
        case 'queues':
            if (window.refreshQueues) window.refreshQueues();
            break;
        case 'events':
            // Events are populated via WebSocket, no refresh needed
            break;
        case 'config':
            if (window.refreshConfig) window.refreshConfig();
            break;
    }
}

function handleEvent(event) {
    // Add to event stream
    if (window.addEvent) {
        window.addEvent(event);
    }

    // Update overview if on that view
    if (currentView === 'overview' && window.refreshOverview) {
        window.refreshOverview();
    }

    // Refresh relevant views based on event type
    if (event.event_type.includes('thunk') && currentView === 'thunks') {
        if (window.refreshThunks) window.refreshThunks();
        // Also update DAG if in DAG view mode
        if (window.currentThunkView === 'dag' && window.thunkDAG) {
            // Try to update just the affected node for smoother updates
            if (event.data && event.data.thunk_id && event.data.status) {
                window.thunkDAG.updateNode(event.data.thunk_id, event.data.status);
            } else if (window.refreshDAG) {
                window.refreshDAG();
            }
        }
    }

    if (event.event_type.includes('agent') && currentView === 'agents') {
        if (window.refreshAgents) window.refreshAgents();
    }

    if (event.event_type.includes('queue') && currentView === 'queues') {
        if (window.refreshQueues) window.refreshQueues();
    }
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', init);
