// Thunk DAG visualization component using D3.js

class ThunkDAG {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.edges = [];
        this.nodeElements = null;
        this.edgeElements = null;
        this.zoom = null;
        this.onNodeClick = null;

        this.config = {
            width: 800,
            height: 500,
            nodeRadius: 20,
            linkDistance: 100,
            chargeStrength: -300,
            statusColors: {
                pending: '#f59e0b',   // amber
                running: '#3b82f6',   // blue
                completed: '#10b981', // green
                success: '#10b981',   // green (alias)
                failed: '#ef4444',    // red
                failure: '#ef4444',   // red (alias)
            },
        };

        this.init();
    }

    init() {
        if (!this.container) return;

        // Clear existing content
        this.container.innerHTML = '';

        // Get container dimensions
        const rect = this.container.getBoundingClientRect();
        this.config.width = rect.width || 800;
        this.config.height = rect.height || 500;

        // Create SVG
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${this.config.width} ${this.config.height}`)
            .attr('class', 'dag-svg');

        // Add zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.svgGroup.attr('transform', event.transform);
            });

        this.svg.call(this.zoom);

        // Create main group for zooming/panning
        this.svgGroup = this.svg.append('g');

        // Add arrow marker for edges
        this.svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('path')
            .attr('d', 'M 0,-5 L 10,0 L 0,5')
            .attr('fill', '#666');

        // Create edge group (edges go behind nodes)
        this.edgeGroup = this.svgGroup.append('g').attr('class', 'edges');

        // Create node group
        this.nodeGroup = this.svgGroup.append('g').attr('class', 'nodes');

        // Create simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(this.config.linkDistance))
            .force('charge', d3.forceManyBody().strength(this.config.chargeStrength))
            .force('center', d3.forceCenter(this.config.width / 2, this.config.height / 2))
            .force('collision', d3.forceCollide().radius(this.config.nodeRadius + 10));
    }

    setData(data) {
        if (!this.svg) return;

        this.nodes = data.nodes || [];
        this.edges = data.edges || [];

        // Update edges
        this.edgeElements = this.edgeGroup
            .selectAll('line')
            .data(this.edges, d => `${d.source}-${d.target}`)
            .join(
                enter => enter.append('line')
                    .attr('class', 'dag-edge')
                    .attr('stroke', '#666')
                    .attr('stroke-width', 2)
                    .attr('marker-end', 'url(#arrowhead)'),
                update => update,
                exit => exit.remove()
            );

        // Update nodes
        const nodeGroups = this.nodeGroup
            .selectAll('g.node')
            .data(this.nodes, d => d.id)
            .join(
                enter => {
                    const g = enter.append('g')
                        .attr('class', 'node')
                        .call(this.drag());

                    // Circle
                    g.append('circle')
                        .attr('r', this.config.nodeRadius)
                        .attr('fill', d => this.getStatusColor(d.status))
                        .attr('stroke', '#fff')
                        .attr('stroke-width', 2)
                        .style('cursor', 'pointer');

                    // Label
                    g.append('text')
                        .attr('dy', 4)
                        .attr('text-anchor', 'middle')
                        .attr('fill', '#fff')
                        .attr('font-size', '10px')
                        .attr('pointer-events', 'none')
                        .text(d => this.truncateLabel(d.label));

                    // Click handler
                    g.on('click', (event, d) => {
                        event.stopPropagation();
                        if (this.onNodeClick) {
                            this.onNodeClick(d);
                        }
                    });

                    return g;
                },
                update => {
                    update.select('circle')
                        .attr('fill', d => this.getStatusColor(d.status));
                    update.select('text')
                        .text(d => this.truncateLabel(d.label));
                    return update;
                },
                exit => exit.remove()
            );

        this.nodeElements = nodeGroups;

        // Update simulation
        this.simulation.nodes(this.nodes);
        this.simulation.force('link').links(this.edges);
        this.simulation.alpha(1).restart();

        // Update positions on tick
        this.simulation.on('tick', () => {
            this.edgeElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            this.nodeElements
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
    }

    updateNode(nodeId, status) {
        // Update a single node's status (for live updates)
        const node = this.nodes.find(n => n.id === nodeId);
        if (node) {
            node.status = status;
            this.nodeGroup
                .selectAll('g.node')
                .filter(d => d.id === nodeId)
                .select('circle')
                .attr('fill', this.getStatusColor(status));
        }
    }

    getStatusColor(status) {
        return this.config.statusColors[status] || this.config.statusColors.pending;
    }

    truncateLabel(label) {
        if (label.length > 8) {
            return label.substring(0, 6) + '..';
        }
        return label;
    }

    drag() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }

    fitToView() {
        if (!this.svg || this.nodes.length === 0) return;

        // Calculate bounds
        const padding = 50;
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        this.nodes.forEach(n => {
            if (n.x < minX) minX = n.x;
            if (n.x > maxX) maxX = n.x;
            if (n.y < minY) minY = n.y;
            if (n.y > maxY) maxY = n.y;
        });

        const width = maxX - minX + padding * 2;
        const height = maxY - minY + padding * 2;
        const midX = (minX + maxX) / 2;
        const midY = (minY + maxY) / 2;

        const scale = Math.min(
            this.config.width / width,
            this.config.height / height,
            1.5
        );

        const translateX = this.config.width / 2 - midX * scale;
        const translateY = this.config.height / 2 - midY * scale;

        this.svg.transition().duration(500)
            .call(
                this.zoom.transform,
                d3.zoomIdentity.translate(translateX, translateY).scale(scale)
            );
    }

    resetZoom() {
        this.svg.transition().duration(500)
            .call(this.zoom.transform, d3.zoomIdentity);
    }

    setNodeClickHandler(handler) {
        this.onNodeClick = handler;
    }

    resize() {
        if (!this.container) return;

        const rect = this.container.getBoundingClientRect();
        this.config.width = rect.width || 800;
        this.config.height = rect.height || 500;

        this.svg
            .attr('viewBox', `0 0 ${this.config.width} ${this.config.height}`);

        this.simulation
            .force('center', d3.forceCenter(this.config.width / 2, this.config.height / 2))
            .alpha(0.3)
            .restart();
    }

    destroy() {
        if (this.simulation) {
            this.simulation.stop();
        }
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

// Export
window.ThunkDAG = ThunkDAG;
