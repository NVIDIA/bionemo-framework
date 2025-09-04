import React, { useMemo } from 'react';
import * as d3 from 'd3';

const SHRINK = 0.85;

const BeamSearch1 = ({ data, width = 1050, height = 120 }) => {
  const margin = { top: 0, right: 100, bottom: 10, left: 155 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const strokeWidth = 2 * SHRINK;

  const DNASequence = ({ sequence = 'ATCGA' }) => {
    const blockWidth = 22 * SHRINK;
    const spacing = 2 * SHRINK;
    const nodeSize = 20 * SHRINK;
    const fillColor = 'var(--inch-worm)';
    const strokeColor = 'var(--deep-fir)';

    return (
      <g transform={`translate(0,0)`}>
        <rect
          x={-nodeSize / 2 - spacing / 2}
          y={-nodeSize / 2}
          width={nodeSize * sequence.length + spacing * sequence.length}
          height={nodeSize}
          fill="white"
          stroke={strokeColor}
          // rx={4}
          strokeWidth={strokeWidth}
        />
        {sequence.split('').map((base, i) => {
          return (
            <g key={i} transform={`translate(${i * blockWidth}, 0)`}>
              <rect
                x={-nodeSize / 2}
                y={-nodeSize / 2}
                width={nodeSize}
                height={nodeSize}
                fill={fillColor}
                rx={0}
                ry={0}
                stroke={strokeColor}
                strokeWidth={strokeWidth}
              />
              <text
                x={0}
                y={3 * SHRINK}
                textAnchor="middle"
                fontSize={`${10 * SHRINK}px`}
                fontWeight="600"
                fill="var(--mine-shaft)"
                style={{ fontFamily: 'Nvidia Sans' }}
              >
                {base}
              </text>
            </g>
          );
        })}
      </g>
    );
  };

  // Calculate node width based on sequence length or fixed size for extensions
  const getNodeWidth = node => {
    if (node.data.isHypothesis) {
      const spacing = 2 * SHRINK;
      const nodeSize = 20 * SHRINK;
      return (
        nodeSize * node.data.sequence.length +
        spacing * node.data.sequence.length
      );
    }
    return 20 * SHRINK; // Fixed width for extension nodes
  };

  // Grid configuration
  const gridOpacity = 0;
  const maxX = 20;
  const maxY = 10;

  const xScale = d3.scaleLinear().domain([0, maxX]).range([0, innerWidth]);

  const yScale = d3.scaleLinear().domain([0, maxY]).range([innerHeight, 0]);

  // Helper functions to create nodes
  const createHypothesisNode = (sequence, active, children) => ({
    name: sequence,
    sequence: sequence,
    isHypothesis: true,
    active: active,
    children: children || [],
  });

  const createExtensionNode = (nucleotide, score, active) => ({
    name: nucleotide,
    isExtension: true,
    score: score,
    active: active,
    children: [],
  });

  // Create the transformed tree structure following the correct pattern
  const createTransformedData = () => {
    const root = {
      name: 'A',
      sequence: 'A',
      isHypothesis: true,
      isFirst: true,
      isPrompt: true,
      active: true,
      children: [
        // First level: S extends to A, B, C
        createExtensionNode('A', -0.39, true),
        createExtensionNode('T', -0.6, false),
        createExtensionNode('C', -0.45, true),
      ],
    };

    // Add A hypothesis after A extension
    root.children[0].children = [createHypothesisNode('AA', true, [])];

    // Add C hypothesis after C extension
    root.children[2].children = [createHypothesisNode('AC', true, [])];

    return root;
  };

  const treeData = data || createTransformedData();

  // Compute tree layout with custom node positioning
  const { root, links } = useMemo(() => {
    const treeLayout = d3
      .tree()
      .size([innerHeight, innerWidth - 100])
      .separation((a, b) => (a.parent === b.parent ? 1.5 : 2));

    const hierarchy = d3.hierarchy(treeData);
    const rootNode = treeLayout(hierarchy);

    // First pass: swap x and y for horizontal layout
    rootNode.descendants().forEach(d => {
      const temp = d.x;
      d.x = d.y + 50;
      d.y = temp;
    });

    // Second pass: adjust x positions to account for node widths
    // Group nodes by depth (x-coordinate after swap)
    const nodesByDepth = {};
    rootNode.descendants().forEach(node => {
      const depth = node.depth;
      if (!nodesByDepth[depth]) {
        nodesByDepth[depth] = [];
      }
      nodesByDepth[depth].push(node);
    });

    // Calculate cumulative x offset for each depth level
    const depthOffsets = {};
    let cumulativeOffset = 50;

    for (
      let depth = 0;
      depth <= Math.max(...Object.keys(nodesByDepth).map(Number));
      depth++
    ) {
      depthOffsets[depth] = cumulativeOffset;

      if (nodesByDepth[depth]) {
        // Find the maximum width at this depth
        const maxWidth = Math.max(
          ...nodesByDepth[depth].map(n => getNodeWidth(n))
        );

        // Add spacing between levels
        const levelSpacing = depth % 2 === 0 ? 80 * SHRINK : 50 * SHRINK;
        cumulativeOffset += maxWidth + levelSpacing;
      }
    }

    // Apply the calculated x positions
    rootNode.descendants().forEach(node => {
      node.x = depthOffsets[node.depth];
    });

    return {
      root: rootNode,
      links: rootNode.links(),
    };
  }, [treeData, innerHeight, innerWidth]);

  // Grid line generators
  const xGridLines = () => {
    return xScale
      .ticks(20)
      .map((tick, i) => (
        <line
          key={`x-grid-${i}`}
          x1={xScale(tick)}
          x2={xScale(tick)}
          y1={yScale(0)}
          y2={yScale(maxY)}
          stroke="black"
          strokeDasharray="2,2"
          opacity={gridOpacity}
        />
      ));
  };

  const yGridLines = () => {
    return yScale
      .ticks(10)
      .map((tick, i) => (
        <line
          key={`y-grid-${i}`}
          x1={xScale(0)}
          x2={xScale(maxX)}
          y1={yScale(tick)}
          y2={yScale(tick)}
          stroke="#e0e0e0"
          strokeDasharray="2,2"
          opacity={gridOpacity}
        />
      ));
  };

  // Render link path (horizontal with right angles)
  const renderLinkPath = link => {
    const sourceXOffset = getNodeWidth(link.source) / 2;
    const targetXOffset = getNodeWidth(link.target) / 2;

    const sourceX = link.source.x + sourceXOffset;
    const targetX = link.target.x - targetXOffset;

    // Special handling for links ending at hypothesis nodes
    // These should point to the left edge of the hypothesis node
    if (link.target.data.isHypothesis) {
      const spacing = 15 * SHRINK;
      const nodeSize = 20 * SHRINK;
      const sequenceStartX = link.target.x - nodeSize / 2 - spacing / 2;

      const midX = (sourceX + sequenceStartX) / 2;

      return `M ${sourceX} ${link.source.y}
              L ${midX} ${link.source.y}
              L ${midX} ${link.target.y}
              L ${sequenceStartX} ${link.target.y}`;
    }

    // Add a little padding before the nonHypothesis final node as well
    const nonHypothesisPadding = 10 * SHRINK;
    const paddedTargetX = targetX - nonHypothesisPadding;
    const midX = (sourceX + paddedTargetX) / 2;

    return `M ${sourceX} ${link.source.y}
            L ${midX} ${link.source.y}
            L ${midX} ${link.target.y}
            L ${paddedTargetX} ${link.target.y}`;
  };

  const renderNode = node => {
    if (node.data.isHypothesis) {
      return (
        <g
          key={`node-${node.depth}-${node.data.name}-${node.x}-${node.y}`}
          transform={`translate(${node.x},${node.y})`}
        >
          <DNASequence sequence={node.data.sequence} />
        </g>
      );
    }

    if (node.data.isFirst) {
      return (
        <g
          key={`node-${node.depth}-${node.data.name}-${node.x}-${node.y}`}
          transform={`translate(${node.x},${node.y})`}
        >
          <DNASequence sequence={node.data.sequence} />
        </g>
      );
    }

    // Render extension nodes with color based on activity
    const nodeSize = 20 * SHRINK;
    const fillColor = node.data.active ? 'var(--pippin)' : 'var(--gray-200)';
    const strokeColor = node.data.active ? 'var(--hot-cinnamon)' : 'black';

    return (
      <g
        key={`node-${node.depth}-${node.data.name}-${node.x}-${node.y}`}
        transform={`translate(${node.x},${node.y})`}
      >
        <rect
          x={-nodeSize / 2}
          y={-nodeSize / 2}
          width={nodeSize}
          height={nodeSize}
          fill={fillColor}
          rx={4 * SHRINK}
          ry={4 * SHRINK}
          stroke={strokeColor}
          strokeWidth={strokeWidth}
        />
        <text
          x={0}
          y={3 * SHRINK}
          textAnchor="middle"
          fontSize={`${10 * SHRINK}px`}
          fontWeight="600"
          fill="#374151"
          style={{ fontFamily: 'Nvidia Sans' }}
        >
          {node.data.name}
        </text>
      </g>
    );
  };

  return (
    <svg
      width={width}
      height={height}
      style={{
        fontFamily: 'Nvidia Sans',
      }}
    >
      {/* Arrow marker definition */}
      <defs>
        <marker
          id="arrow"
          viewBox="0 -5 10 10"
          refX={8 * SHRINK}
          refY={0}
          markerWidth={5 * SHRINK}
          markerHeight={5 * SHRINK}
          orient="auto"
        >
          <path d="M0,-5L10,0L0,5" fill="var(--mine-shaft)" />
        </marker>
      </defs>

      <g transform={`translate(${margin.left},${margin.top})`}>
        {/* Grid lines - render first so they're behind everything */}
        {xGridLines()}
        {yGridLines()}

        {/* Links */}
        {links.map((link, i) => (
          <g key={`link-${i}`}>
            <path
              d={renderLinkPath(link)}
              stroke="var(--silver-chalice)"
              strokeWidth={1 * SHRINK}
              fill="none"
              markerEnd="url(#arrow)"
            />
          </g>
        ))}

        {/* Nodes */}
        {root.descendants().map(node => renderNode(node))}
      </g>
    </svg>
  );
};

export default BeamSearch1;
