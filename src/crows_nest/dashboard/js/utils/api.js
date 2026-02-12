// API Client for Crow's Nest Dashboard

const API_BASE = '/api/dashboard';

const api = {
    async get(endpoint) {
        const response = await fetch(`${API_BASE}${endpoint}`);
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        return response.json();
    },

    // Overview
    async getOverview() {
        return this.get('/overview');
    },

    // Thunks
    async getThunks(status = null, limit = 50, offset = 0) {
        let url = `/thunks?limit=${limit}&offset=${offset}`;
        if (status) {
            url += `&status=${status}`;
        }
        return this.get(url);
    },

    async getThunkDetail(thunkId) {
        return this.get(`/thunks/${thunkId}`);
    },

    async getThunkDAG(limit = 100, traceId = null) {
        let url = `/thunks/graph?limit=${limit}`;
        if (traceId) {
            url += `&trace_id=${encodeURIComponent(traceId)}`;
        }
        return this.get(url);
    },

    // Agents
    async getAgents(includeTerminated = false) {
        return this.get(`/agents?include_terminated=${includeTerminated}`);
    },

    async getAgentTree(agentId) {
        return this.get(`/agents/${agentId}/tree`);
    },

    // Queues
    async getQueues() {
        return this.get('/queues');
    },

    // Memory
    async getMemory(agentId, query = null, memoryType = null, limit = 20) {
        let url = `/memory/${agentId}?limit=${limit}`;
        if (query) url += `&query=${encodeURIComponent(query)}`;
        if (memoryType) url += `&memory_type=${memoryType}`;
        return this.get(url);
    },

    // Config
    async getConfig() {
        return this.get('/config');
    }
};

// Export for use in other modules
window.api = api;
