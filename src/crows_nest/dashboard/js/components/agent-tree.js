// Agent tree component

let currentAgents = [];
let selectedAgentId = null;

async function refreshAgents() {
    try {
        const includeTerminated = document.getElementById('show-terminated')?.checked || false;
        currentAgents = await api.getAgents(includeTerminated);
        renderAgentTree();
    } catch (e) {
        console.error('Failed to refresh agents:', e);
    }
}

function renderAgentTree() {
    const container = document.getElementById('agent-tree');
    if (!container) return;

    if (currentAgents.length === 0) {
        container.innerHTML = '<p class="placeholder">No agents found</p>';
        return;
    }

    // Build tree structure
    const rootAgents = currentAgents.filter(a => !a.parent_id);
    const childMap = {};
    currentAgents.forEach(agent => {
        if (agent.parent_id) {
            if (!childMap[agent.parent_id]) {
                childMap[agent.parent_id] = [];
            }
            childMap[agent.parent_id].push(agent);
        }
    });

    function renderNode(agent, isRoot = false) {
        const children = childMap[agent.id] || [];
        const childrenHtml = children.map(child => renderNode(child)).join('');

        return `
            <div class="tree-node ${isRoot ? 'root' : ''}">
                <div class="tree-node-content ${agent.is_active ? '' : 'inactive'}"
                     onclick="selectAgent('${agent.id}')">
                    ${agent.id.substring(0, 8)}...
                    ${!agent.is_active ? ' (terminated)' : ''}
                </div>
                ${childrenHtml}
            </div>
        `;
    }

    container.innerHTML = rootAgents.map(agent => renderNode(agent, true)).join('');
}

async function selectAgent(agentId) {
    selectedAgentId = agentId;

    const detailContainer = document.getElementById('agent-detail');
    if (!detailContainer) return;

    try {
        const tree = await api.getAgentTree(agentId);
        renderAgentDetail(tree);
    } catch (e) {
        // Fallback to basic info from list
        const agent = currentAgents.find(a => a.id === agentId);
        if (agent) {
            renderAgentDetailBasic(agent);
        } else {
            detailContainer.innerHTML = `<p class="placeholder">Failed to load agent details</p>`;
        }
    }
}

function renderAgentDetail(tree) {
    const container = document.getElementById('agent-detail');
    if (!container) return;

    container.innerHTML = `
        <div class="detail-section">
            <h4>Agent ID</h4>
            <div class="detail-content">${tree.id}</div>
        </div>
        <div class="detail-section">
            <h4>Status</h4>
            <span class="thunk-status ${tree.is_active ? 'success' : 'failure'}">
                ${tree.is_active ? 'Active' : 'Terminated'}
            </span>
        </div>
        <div class="detail-section">
            <h4>Depth</h4>
            <div class="detail-content">${tree.depth}</div>
        </div>
        <div class="detail-section">
            <h4>Capabilities</h4>
            <div class="detail-content">${tree.capabilities.join('\n') || 'None'}</div>
        </div>
        <div class="detail-section">
            <h4>Children</h4>
            <div class="detail-content">${tree.children.length} child agent(s)</div>
        </div>
        ${tree.children.length > 0 ? `
        <div class="detail-section">
            <h4>Child IDs</h4>
            <div class="detail-content">${tree.children.map(c => c.id.substring(0, 8) + '...').join('\n')}</div>
        </div>
        ` : ''}
    `;
}

function renderAgentDetailBasic(agent) {
    const container = document.getElementById('agent-detail');
    if (!container) return;

    container.innerHTML = `
        <div class="detail-section">
            <h4>Agent ID</h4>
            <div class="detail-content">${agent.id}</div>
        </div>
        <div class="detail-section">
            <h4>Status</h4>
            <span class="thunk-status ${agent.is_active ? 'success' : 'failure'}">
                ${agent.is_active ? 'Active' : 'Terminated'}
            </span>
        </div>
        <div class="detail-section">
            <h4>Parent</h4>
            <div class="detail-content">${agent.parent_id || 'None (root agent)'}</div>
        </div>
        <div class="detail-section">
            <h4>Depth</h4>
            <div class="detail-content">${agent.depth}</div>
        </div>
        <div class="detail-section">
            <h4>Children</h4>
            <div class="detail-content">${agent.children_count} child agent(s)</div>
        </div>
        <div class="detail-section">
            <h4>Created</h4>
            <div class="detail-content">${new Date(agent.created_at).toLocaleString()}</div>
        </div>
    `;
}

// Export functions
window.refreshAgents = refreshAgents;
window.selectAgent = selectAgent;
