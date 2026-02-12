// Config view component

let currentConfig = null;

async function refreshConfig() {
    try {
        currentConfig = await api.getConfig();
        renderConfig();
    } catch (e) {
        console.error('Failed to refresh config:', e);
    }
}

function renderConfig() {
    const container = document.getElementById('config-display');
    if (!container) return;

    if (!currentConfig) {
        container.innerHTML = '<p class="placeholder">Failed to load configuration</p>';
        return;
    }

    const configItems = [
        { key: 'Database Path', value: currentConfig.database_path },
        { key: 'API Host', value: currentConfig.api_host },
        { key: 'API Port', value: currentConfig.api_port },
        { key: 'Log Level', value: currentConfig.log_level },
        { key: 'Tracing Enabled', value: currentConfig.tracing_enabled ? 'Yes' : 'No' },
    ];

    container.innerHTML = configItems.map(item => `
        <div class="config-item">
            <span class="config-key">${item.key}</span>
            <span class="config-value">${item.value}</span>
        </div>
    `).join('');
}

// Export functions
window.refreshConfig = refreshConfig;
