// Thunk list and detail component

let currentThunks = [];
let selectedThunkId = null;
let currentThunkView = 'list';
let thunkDAG = null;

function switchThunkView(view) {
    currentThunkView = view;

    // Update tab buttons
    document.querySelectorAll('.view-tabs .tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === view);
    });

    // Update view content
    document.getElementById('thunk-list-view')?.classList.toggle('active', view === 'list');
    document.getElementById('thunk-dag-view')?.classList.toggle('active', view === 'dag');

    // Initialize or refresh DAG when switching to DAG view
    if (view === 'dag') {
        refreshDAG();
    }
}

async function refreshDAG() {
    try {
        const data = await api.getThunkDAG(100);

        if (!thunkDAG) {
            thunkDAG = new ThunkDAG('thunk-dag-container');
            thunkDAG.setNodeClickHandler(handleDAGNodeClick);
        }

        thunkDAG.setData(data);

        // Auto-fit after initial load
        setTimeout(() => thunkDAG.fitToView(), 500);
    } catch (e) {
        console.error('Failed to refresh DAG:', e);
    }
}

function handleDAGNodeClick(node) {
    const detailContainer = document.getElementById('dag-node-detail');
    if (!detailContainer) return;

    // Fetch full thunk details
    api.getThunkDetail(node.id).then(thunk => {
        renderDAGNodeDetail(thunk, detailContainer);
    }).catch(() => {
        detailContainer.innerHTML = '<p class="placeholder">Failed to load details</p>';
    });
}

function renderDAGNodeDetail(thunk, container) {
    container.innerHTML = `
        <div class="detail-section">
            <h4>Operation</h4>
            <div class="detail-content">${thunk.operation}</div>
        </div>
        <div class="detail-section">
            <h4>Status</h4>
            <span class="thunk-status ${thunk.status}">${thunk.status}</span>
            ${thunk.duration_ms ? ` (${thunk.duration_ms}ms)` : ''}
        </div>
        <div class="detail-section">
            <h4>ID</h4>
            <div class="detail-content" style="font-size: 0.75rem;">${thunk.id}</div>
        </div>
        ${thunk.dependencies.length > 0 ? `
        <div class="detail-section">
            <h4>Dependencies</h4>
            <div class="detail-content" style="font-size: 0.75rem;">${thunk.dependencies.join('\n')}</div>
        </div>
        ` : ''}
        ${thunk.value !== null ? `
        <div class="detail-section">
            <h4>Result</h4>
            <div class="detail-content">${JSON.stringify(thunk.value, null, 2)}</div>
        </div>
        ` : ''}
        ${thunk.error ? `
        <div class="detail-section">
            <h4>Error</h4>
            <div class="detail-content" style="color: var(--error);">${JSON.stringify(thunk.error, null, 2)}</div>
        </div>
        ` : ''}
    `;
}

function fitDAGToView() {
    if (thunkDAG) {
        thunkDAG.fitToView();
    }
}

function resetDAGZoom() {
    if (thunkDAG) {
        thunkDAG.resetZoom();
    }
}

async function refreshThunks() {
    try {
        const filter = document.getElementById('thunk-filter')?.value || null;
        currentThunks = await api.getThunks(filter);
        renderThunkList();
    } catch (e) {
        console.error('Failed to refresh thunks:', e);
    }
}

async function filterThunks() {
    await refreshThunks();
}

function renderThunkList() {
    const container = document.getElementById('thunk-list');
    if (!container) return;

    if (currentThunks.length === 0) {
        container.innerHTML = '<p class="placeholder">No thunks found</p>';
        return;
    }

    container.innerHTML = currentThunks.map(thunk => `
        <div class="thunk-item ${selectedThunkId === thunk.id ? 'selected' : ''}"
             onclick="selectThunk('${thunk.id}')">
            <div class="thunk-operation">${thunk.operation}</div>
            <div class="thunk-id">${thunk.id}</div>
            <span class="thunk-status ${thunk.status}">${thunk.status}</span>
            ${thunk.duration_ms ? `<span class="thunk-duration">${thunk.duration_ms}ms</span>` : ''}
        </div>
    `).join('');
}

async function selectThunk(thunkId) {
    selectedThunkId = thunkId;
    renderThunkList();

    const detailContainer = document.getElementById('thunk-detail');
    if (!detailContainer) return;

    try {
        const thunk = await api.getThunkDetail(thunkId);
        renderThunkDetail(thunk);
    } catch (e) {
        detailContainer.innerHTML = `<p class="placeholder">Failed to load thunk details</p>`;
    }
}

function renderThunkDetail(thunk) {
    const container = document.getElementById('thunk-detail');
    if (!container) return;

    container.innerHTML = `
        <div class="detail-section">
            <h4>Operation</h4>
            <div class="detail-content">${thunk.operation}</div>
        </div>
        <div class="detail-section">
            <h4>Status</h4>
            <span class="thunk-status ${thunk.status}">${thunk.status}</span>
            ${thunk.duration_ms ? ` (${thunk.duration_ms}ms)` : ''}
        </div>
        <div class="detail-section">
            <h4>ID</h4>
            <div class="detail-content">${thunk.id}</div>
        </div>
        <div class="detail-section">
            <h4>Trace ID</h4>
            <div class="detail-content">${thunk.trace_id}</div>
        </div>
        <div class="detail-section">
            <h4>Created</h4>
            <div class="detail-content">${new Date(thunk.created_at).toLocaleString()}</div>
        </div>
        ${thunk.forced_at ? `
        <div class="detail-section">
            <h4>Executed</h4>
            <div class="detail-content">${new Date(thunk.forced_at).toLocaleString()}</div>
        </div>
        ` : ''}
        <div class="detail-section">
            <h4>Inputs</h4>
            <div class="detail-content">${JSON.stringify(thunk.inputs, null, 2)}</div>
        </div>
        <div class="detail-section">
            <h4>Capabilities</h4>
            <div class="detail-content">${thunk.capabilities.join(', ') || 'None'}</div>
        </div>
        ${thunk.dependencies.length > 0 ? `
        <div class="detail-section">
            <h4>Dependencies</h4>
            <div class="detail-content">${thunk.dependencies.join('\n')}</div>
        </div>
        ` : ''}
        ${thunk.value !== null ? `
        <div class="detail-section">
            <h4>Result</h4>
            <div class="detail-content">${JSON.stringify(thunk.value, null, 2)}</div>
        </div>
        ` : ''}
        ${thunk.error ? `
        <div class="detail-section">
            <h4>Error</h4>
            <div class="detail-content" style="color: var(--error);">${JSON.stringify(thunk.error, null, 2)}</div>
        </div>
        ` : ''}
    `;
}

// Export functions and state
window.refreshThunks = refreshThunks;
window.filterThunks = filterThunks;
window.selectThunk = selectThunk;
window.switchThunkView = switchThunkView;
window.refreshDAG = refreshDAG;
window.fitDAGToView = fitDAGToView;
window.resetDAGZoom = resetDAGZoom;

// Export state getters for app.js to check DAG view status
Object.defineProperty(window, 'currentThunkView', {
    get: () => currentThunkView
});
Object.defineProperty(window, 'thunkDAG', {
    get: () => thunkDAG
});
