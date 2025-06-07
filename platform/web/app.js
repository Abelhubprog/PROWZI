
/**
 * Prowzi Agent Platform - Frontend Application
 * Real-time dashboard for monitoring autonomous AI agents
 */

class ProwziApp {
    constructor() {
        this.briefs = [];
        this.agents = [];
        this.events = [];
        this.missions = [];
        this.filteredBriefs = [];
        this.briefSearchTerm = '';
        this.agentSearchTerm = '';
        this.selectedAgentId = null;
        this.sortBy = 'creationDate';
        this.sortOrder = 'desc';
        this.loading = true;
        this.error = null;
        this.ws = null;

        this.init();
    }

    async init() {
        this.setupWebSocket();
        await this.fetchInitialData();
        this.attachEventListeners();
        this.render();
        this.startPeriodicUpdates();
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            // Reconnect after 3 seconds
            setTimeout(() => this.setupWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'agent_update':
                this.updateAgent(data.agent);
                break;
            case 'new_brief':
                this.addBrief(data.brief);
                break;
            case 'new_event':
                this.addEvent(data.event);
                break;
            case 'mission_update':
                this.updateMission(data.mission);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    async fetchInitialData() {
        this.loading = true;
        this.error = null;

        try {
            const [agentsResponse, briefsResponse, eventsResponse, missionsResponse] = await Promise.all([
                this.fetchWithFallback('/api/v1/agents'),
                this.fetchWithFallback('/api/v1/briefs'),
                this.fetchWithFallback('/api/v1/events'),
                this.fetchWithFallback('/api/v1/missions')
            ]);

            this.agents = agentsResponse;
            this.briefs = briefsResponse;
            this.events = eventsResponse;
            this.missions = missionsResponse;
            this.filteredBriefs = [...this.briefs];

        } catch (error) {
            console.error('Error fetching data:', error);
            this.error = 'Failed to load data. Please refresh the page.';
        } finally {
            this.loading = false;
        }
    }

    async fetchWithFallback(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                console.log(`Fallback for ${url}:`, `HTTP ${response.status}`);
                return [];
            }
            return await response.json();
        } catch (error) {
            console.log(`Fallback for ${url}:`, error.message);
            return [];
        }
    }

    updateAgent(agent) {
        const index = this.agents.findIndex(a => a.id === agent.id);
        if (index !== -1) {
            this.agents[index] = agent;
        } else {
            this.agents.push(agent);
        }
        this.renderAgents();
    }

    addBrief(brief) {
        this.briefs.unshift(brief);
        this.filterBriefs();
        this.renderBriefs();
    }

    addEvent(event) {
        this.events.unshift(event);
        this.renderEvents();
    }

    updateMission(mission) {
        const index = this.missions.findIndex(m => m.id === mission.id);
        if (index !== -1) {
            this.missions[index] = mission;
        } else {
            this.missions.push(mission);
        }
        this.renderMissions();
    }

    attachEventListeners() {
        // Navigation
        const navButtons = document.querySelectorAll('.nav-btn');
        if (navButtons) {
            navButtons.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    this.switchTab(e.target.dataset.tab);
                });
            });
        }

        // Search inputs
        const briefSearchInput = document.getElementById('brief-search');
        if (briefSearchInput) {
            briefSearchInput.addEventListener('input', (e) => {
                this.briefSearchTerm = e.target.value;
                this.filterBriefs();
            });
        }

        const agentSearchInput = document.getElementById('agent-search');
        if (agentSearchInput) {
            agentSearchInput.addEventListener('input', (e) => {
                this.agentSearchTerm = e.target.value;
                this.renderAgents();
            });
        }

        // Sort controls
        const sortSelect = document.getElementById('sort-select');
        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => {
                this.sortBy = e.target.value;
                this.sortBriefs();
            });
        }

        const sortOrderSelect = document.getElementById('sort-order');
        if (sortOrderSelect) {
            sortOrderSelect.addEventListener('change', (e) => {
                this.sortOrder = e.target.value;
                this.sortBriefs();
            });
        }

        // Create mission button
        const createMissionBtn = document.getElementById('create-mission-btn');
        if (createMissionBtn) {
            createMissionBtn.addEventListener('click', () => {
                this.showCreateMissionModal();
            });
        }
    }

    switchTab(tabName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Show/hide content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    filterBriefs() {
        this.filteredBriefs = this.briefs.filter(brief => {
            const searchTerm = this.briefSearchTerm.toLowerCase();
            return brief.headline.toLowerCase().includes(searchTerm) ||
                   brief.content.summary.toLowerCase().includes(searchTerm);
        });
        this.sortBriefs();
    }

    sortBriefs() {
        this.filteredBriefs.sort((a, b) => {
            let aValue, bValue;
            
            switch (this.sortBy) {
                case 'impact':
                    aValue = this.getImpactScore(a.impactLevel);
                    bValue = this.getImpactScore(b.impactLevel);
                    break;
                case 'confidence':
                    aValue = a.confidenceScore;
                    bValue = b.confidenceScore;
                    break;
                default:
                    aValue = new Date(a.createdAt);
                    bValue = new Date(b.createdAt);
            }

            if (this.sortOrder === 'desc') {
                return bValue > aValue ? 1 : -1;
            } else {
                return aValue > bValue ? 1 : -1;
            }
        });
        this.renderBriefs();
    }

    getImpactScore(level) {
        const scores = { critical: 4, high: 3, medium: 2, low: 1 };
        return scores[level] || 0;
    }

    render() {
        this.renderDashboard();
        this.renderAgents();
        this.renderBriefs();
        this.renderEvents();
        this.renderMissions();
    }

    renderDashboard() {
        const totalAgents = this.agents.length;
        const activeAgents = this.agents.filter(a => a.status === 'running').length;
        const totalBriefs = this.briefs.length;
        const activeMissions = this.missions.filter(m => m.status === 'active').length;

        document.getElementById('total-agents').textContent = totalAgents;
        document.getElementById('active-agents').textContent = activeAgents;
        document.getElementById('total-briefs').textContent = totalBriefs;
        document.getElementById('active-missions').textContent = activeMissions;
    }

    renderAgents() {
        const container = document.getElementById('agents-list');
        if (!container) return;

        const filteredAgents = this.agents.filter(agent => {
            const searchTerm = this.agentSearchTerm.toLowerCase();
            return agent.name.toLowerCase().includes(searchTerm) ||
                   agent.type.toLowerCase().includes(searchTerm);
        });

        container.innerHTML = filteredAgents.map(agent => `
            <div class="agent-card ${agent.status}">
                <div class="agent-header">
                    <h3>${agent.name}</h3>
                    <span class="status-badge ${agent.status}">${agent.status}</span>
                </div>
                <div class="agent-details">
                    <p><strong>Type:</strong> ${agent.type}</p>
                    <p><strong>Mission:</strong> ${agent.mission_id || 'None'}</p>
                    <p><strong>Created:</strong> ${new Date(agent.created_at).toLocaleString()}</p>
                </div>
                ${this.renderMetrics(agent.metrics)}
            </div>
        `).join('');
    }

    renderMetrics(metrics) {
        if (!metrics || Object.keys(metrics).length === 0) {
            return '<div class="metrics">No metrics available</div>';
        }

        return `
            <div class="metrics">
                <h4>Metrics</h4>
                ${Object.entries(metrics).map(([key, value]) => `
                    <div class="metric">
                        <span class="metric-name">${key}:</span>
                        <span class="metric-value">${value}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    renderBriefs() {
        const container = document.getElementById('briefs-list');
        if (!container) return;

        if (this.loading) {
            container.innerHTML = '<div class="loading">Loading briefs...</div>';
            return;
        }

        if (this.filteredBriefs.length === 0) {
            container.innerHTML = '<div class="empty-state">No briefs found</div>';
            return;
        }

        container.innerHTML = this.filteredBriefs.map(brief => `
            <div class="brief-card ${brief.impactLevel}">
                <div class="brief-header">
                    <h3>${brief.headline}</h3>
                    <div class="brief-meta">
                        <span class="impact-badge ${brief.impactLevel}">${brief.impactLevel}</span>
                        <span class="confidence">Confidence: ${(brief.confidenceScore * 100).toFixed(1)}%</span>
                        <span class="timestamp">${new Date(brief.createdAt).toLocaleString()}</span>
                    </div>
                </div>
                <div class="brief-content">
                    <p>${brief.content.summary}</p>
                    ${brief.content.evidence ? `
                        <div class="evidence">
                            <h4>Evidence</h4>
                            <ul>
                                ${brief.content.evidence.map(e => `<li>${e.text} (${e.confidence})</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    ${brief.content.suggestedActions ? `
                        <div class="actions">
                            <h4>Suggested Actions</h4>
                            <ul>
                                ${brief.content.suggestedActions.map(action => `<li>${action}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }

    renderEvents() {
        const container = document.getElementById('events-list');
        if (!container) return;

        container.innerHTML = this.events.slice(0, 50).map(event => `
            <div class="event-item">
                <div class="event-header">
                    <span class="event-type">${event.type}</span>
                    <span class="event-timestamp">${new Date(event.timestamp).toLocaleString()}</span>
                </div>
                <div class="event-source">Source: ${event.source}</div>
                <div class="event-data">${JSON.stringify(event.data, null, 2)}</div>
            </div>
        `).join('');
    }

    renderMissions() {
        const container = document.getElementById('missions-list');
        if (!container) return;

        container.innerHTML = this.missions.map(mission => `
            <div class="mission-card ${mission.status}">
                <div class="mission-header">
                    <h3>${mission.name}</h3>
                    <span class="status-badge ${mission.status}">${mission.status}</span>
                </div>
                <div class="mission-details">
                    <p><strong>Created:</strong> ${new Date(mission.created_at).toLocaleString()}</p>
                    <div class="mission-config">
                        <pre>${JSON.stringify(mission.config, null, 2)}</pre>
                    </div>
                </div>
            </div>
        `).join('');
    }

    showCreateMissionModal() {
        // Implementation for mission creation modal
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <h2>Create New Mission</h2>
                <form id="create-mission-form">
                    <div class="form-group">
                        <label for="mission-name">Mission Name:</label>
                        <input type="text" id="mission-name" required>
                    </div>
                    <div class="form-group">
                        <label for="mission-prompt">Prompt:</label>
                        <textarea id="mission-prompt" rows="4" required></textarea>
                    </div>
                    <div class="form-actions">
                        <button type="button" onclick="this.closest('.modal').remove()">Cancel</button>
                        <button type="submit">Create Mission</button>
                    </div>
                </form>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        const form = modal.querySelector('#create-mission-form');
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.createMission({
                name: document.getElementById('mission-name').value,
                prompt: document.getElementById('mission-prompt').value
            });
            modal.remove();
        });
    }

    async createMission(missionData) {
        try {
            const response = await fetch('/api/v1/missions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(missionData)
            });

            if (response.ok) {
                const mission = await response.json();
                this.missions.push(mission);
                this.renderMissions();
            } else {
                throw new Error('Failed to create mission');
            }
        } catch (error) {
            console.error('Error creating mission:', error);
            alert('Failed to create mission. Please try again.');
        }
    }

    startPeriodicUpdates() {
        setInterval(async () => {
            try {
                const agents = await this.fetchWithFallback('/api/v1/agents');
                this.agents = agents;
                this.renderAgents();
                this.renderDashboard();
            } catch (error) {
                console.error('Error updating agents:', error);
            }
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.prowziApp = new ProwziApp();
});

// Export for global access
window.app = window.prowziApp;
