import React, { useRef, useState } from 'react';
import * as d3 from 'd3';

const HyenaOperatorDiagram = () => {
  const svgRef = useRef();
  const width = 850;
  const height = 750;
  const MARGIN = { top: 50, right: 50, bottom: 50, left: 50 };
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;
  const strokeColor = 'var(--silver-chalice)';
  const strokeWidth = 1;
  const gridOpacity = 0.2;
  const maxX = 9;
  const maxY = 10;

  // State for hover interaction
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [verticalOffsetIndex, setVerticalOffsetIndex] = useState(null);
  const [hoveredSeqIndex, setHoveredSeqIndex] = useState(null);

  // Data arrays
  const arrowLines = [
    { x1: 7, y1: 8, x2: 7, y2: 3.25 },
    { x1: 7, y1: 3, x2: 7, y2: 2.4 },
    { x1: 5, y1: 10, x2: 5, y2: 7.3 },
    { x1: 3, y1: 8, x2: 3, y2: 7.3 },
    { x1: 5, y1: 7, x2: 5, y2: 6.4 },
    { x1: 3, y1: 7, x2: 3, y2: 6.4 },
    { x1: 4, y1: 5, x2: 4, y2: 4.4 },
    { x1: 4, y1: 4, x2: 4, y2: 3.25 },
    { x1: 4, y1: 3, x2: 4, y2: 2.45 },
    { x1: 5.5, y1: 1, x2: 5.5, y2: 0.4 },
  ];

  const curvedArrowLines = [
    { x1: 5, y1: 10, x2: 3, y2: 8 },
    { x1: 5, y1: 10, x2: 7, y2: 8 },
    //
    { x1: 3, y1: 5.8, x2: 4, y2: 5.15 },
    { x1: 5, y1: 5.8, x2: 4, y2: 5.15 },
    { x1: 4, y1: 1.8, x2: 5.5, y2: 1.1 },
    { x1: 7, y1: 1.8, x2: 5.5, y2: 1.1 },
  ];

  const dnaSeqs = [
    { x: 5, y: 10, label: '' }, // input
    { x: 3, y: 8, label: 'q' }, // q
    { x: 5, y: 8, label: 'k' }, // k
    { x: 7, y: 8, label: 'v' }, // v
    { x: 3, y: 6, label: '', histogramHeights: [0.65, 0.45, 0.75, 0.25, 0.55] }, // q after conv
    { x: 5, y: 6, label: '', histogramHeights: [0.85, 0.65, 0.55, 0.45, 0.75] }, // k after conv
    { x: 4, y: 4, label: '', histogramHeights: [0.55, 0.29, 0.41, 0.11, 0.41] }, // q·k (elementwise product)
    { x: 4, y: 2, label: '', histogramHeights: [0.5, 0.3, 0.4, 0.1, 0.4] }, // after S/M/L Hyena
    { x: 7, y: 2, label: '', histogramHeights: [0.55, 0.75, 0.65, 0.55, 0.85] }, // v after conv
    {
      x: 5.5,
      y: 0,
      label: '',
      histogramHeights: [0.28, 0.23, 0.26, 0.06, 0.34],
    }, // final output (gated)
  ];

  const operations = [
    { x: 5, y: 7, label: 'Short Explicit Conv' },
    { x: 3, y: 7, label: 'Short Explicit Conv' },
    { x: 4, y: 5, label: 'Elementwise Gate' },
    { x: 4, y: 3, label: '[SE/MR/LI] Convolution' },
    { x: 7, y: 3, label: 'Short Explicit Conv' },
    { x: 5.5, y: 1, label: 'Elementwise Gate' },
  ];

  const textAnnotations = [
    {
      x: -0.5,
      y: 9,
      text: 'First, the input batch of sequences is projected into 3 dense vectors, q, k, and v.',
    },
    {
      x: -0.5,
      y: 7,
      text: 'Next, both q and k are each passed through a short explicit convolution.',
    },
    {
      x: -0.5,
      y: 5,
      text: 'Then, to figure out which keys the query should pay attention to, we compute an elementwise product between q and k.',
    },
    {
      x: -0.5,
      y: 3,
      text: 'The result is then passed through one of the three convolutions, short explicit, medium regularized, or long implicit.',
    },
    {
      x: 7.75,
      y: 4.2,
      text: 'The v projection is also passed through a short explicit convolution to weight the vector with short-term context.',
      chunkSize: 4,
    },
    {
      x: -0.5,
      y: 1,
      text: 'Finally, the output of the convolution above operator is gated with v to produce the final weighted context representation for this layer.',
    },
  ];

  // Sankey link definitions - define which nodes connect to which
  const sankeyLinks = [
    // From input (seq 0) to q, k, v (seq 1, 2, 3) - direct 1:1 mapping
    {
      source: { seqIndex: 0, nodeIndex: 0 },
      target: { seqIndex: 1, nodeIndex: 0 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 1 },
      target: { seqIndex: 1, nodeIndex: 1 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 2 },
      target: { seqIndex: 1, nodeIndex: 2 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 3 },
      target: { seqIndex: 1, nodeIndex: 3 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 4 },
      target: { seqIndex: 1, nodeIndex: 4 },
      value: 1.0,
    },

    {
      source: { seqIndex: 0, nodeIndex: 0 },
      target: { seqIndex: 2, nodeIndex: 0 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 1 },
      target: { seqIndex: 2, nodeIndex: 1 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 2 },
      target: { seqIndex: 2, nodeIndex: 2 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 3 },
      target: { seqIndex: 2, nodeIndex: 3 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 4 },
      target: { seqIndex: 2, nodeIndex: 4 },
      value: 1.0,
    },

    {
      source: { seqIndex: 0, nodeIndex: 0 },
      target: { seqIndex: 3, nodeIndex: 0 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 1 },
      target: { seqIndex: 3, nodeIndex: 1 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 2 },
      target: { seqIndex: 3, nodeIndex: 2 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 3 },
      target: { seqIndex: 3, nodeIndex: 3 },
      value: 1.0,
    },
    {
      source: { seqIndex: 0, nodeIndex: 4 },
      target: { seqIndex: 3, nodeIndex: 4 },
      value: 1.0,
    },

    // From q (seq 1) through Short Explicit Conv to convolved q (seq 4)
    // Each output connects to all inputs (convolution effect)
    {
      source: { seqIndex: 1, nodeIndex: 0 },
      target: { seqIndex: 4, nodeIndex: 0 },
      value: 0.4,
    },
    {
      source: { seqIndex: 1, nodeIndex: 1 },
      target: { seqIndex: 4, nodeIndex: 0 },
      value: 0.3,
    },
    {
      source: { seqIndex: 1, nodeIndex: 2 },
      target: { seqIndex: 4, nodeIndex: 0 },
      value: 0.3,
    },

    {
      source: { seqIndex: 1, nodeIndex: 0 },
      target: { seqIndex: 4, nodeIndex: 1 },
      value: 0.2,
    },
    {
      source: { seqIndex: 1, nodeIndex: 1 },
      target: { seqIndex: 4, nodeIndex: 1 },
      value: 0.5,
    },
    {
      source: { seqIndex: 1, nodeIndex: 2 },
      target: { seqIndex: 4, nodeIndex: 1 },
      value: 0.3,
    },

    {
      source: { seqIndex: 1, nodeIndex: 1 },
      target: { seqIndex: 4, nodeIndex: 2 },
      value: 0.3,
    },
    {
      source: { seqIndex: 1, nodeIndex: 2 },
      target: { seqIndex: 4, nodeIndex: 2 },
      value: 0.4,
    },
    {
      source: { seqIndex: 1, nodeIndex: 3 },
      target: { seqIndex: 4, nodeIndex: 2 },
      value: 0.3,
    },

    {
      source: { seqIndex: 1, nodeIndex: 2 },
      target: { seqIndex: 4, nodeIndex: 3 },
      value: 0.2,
    },
    {
      source: { seqIndex: 1, nodeIndex: 3 },
      target: { seqIndex: 4, nodeIndex: 3 },
      value: 0.4,
    },
    {
      source: { seqIndex: 1, nodeIndex: 4 },
      target: { seqIndex: 4, nodeIndex: 3 },
      value: 0.4,
    },

    {
      source: { seqIndex: 1, nodeIndex: 3 },
      target: { seqIndex: 4, nodeIndex: 4 },
      value: 0.3,
    },
    {
      source: { seqIndex: 1, nodeIndex: 4 },
      target: { seqIndex: 4, nodeIndex: 4 },
      value: 0.7,
    },

    // From k (seq 2) through Short Explicit Conv to convolved k (seq 5)
    {
      source: { seqIndex: 2, nodeIndex: 0 },
      target: { seqIndex: 5, nodeIndex: 0 },
      value: 0.5,
    },
    {
      source: { seqIndex: 2, nodeIndex: 1 },
      target: { seqIndex: 5, nodeIndex: 0 },
      value: 0.3,
    },
    {
      source: { seqIndex: 2, nodeIndex: 2 },
      target: { seqIndex: 5, nodeIndex: 0 },
      value: 0.2,
    },

    {
      source: { seqIndex: 2, nodeIndex: 0 },
      target: { seqIndex: 5, nodeIndex: 1 },
      value: 0.3,
    },
    {
      source: { seqIndex: 2, nodeIndex: 1 },
      target: { seqIndex: 5, nodeIndex: 1 },
      value: 0.4,
    },
    {
      source: { seqIndex: 2, nodeIndex: 2 },
      target: { seqIndex: 5, nodeIndex: 1 },
      value: 0.3,
    },

    {
      source: { seqIndex: 2, nodeIndex: 1 },
      target: { seqIndex: 5, nodeIndex: 2 },
      value: 0.2,
    },
    {
      source: { seqIndex: 2, nodeIndex: 2 },
      target: { seqIndex: 5, nodeIndex: 2 },
      value: 0.5,
    },
    {
      source: { seqIndex: 2, nodeIndex: 3 },
      target: { seqIndex: 5, nodeIndex: 2 },
      value: 0.3,
    },

    {
      source: { seqIndex: 2, nodeIndex: 2 },
      target: { seqIndex: 5, nodeIndex: 3 },
      value: 0.3,
    },
    {
      source: { seqIndex: 2, nodeIndex: 3 },
      target: { seqIndex: 5, nodeIndex: 3 },
      value: 0.4,
    },
    {
      source: { seqIndex: 2, nodeIndex: 4 },
      target: { seqIndex: 5, nodeIndex: 3 },
      value: 0.3,
    },

    {
      source: { seqIndex: 2, nodeIndex: 3 },
      target: { seqIndex: 5, nodeIndex: 4 },
      value: 0.4,
    },
    {
      source: { seqIndex: 2, nodeIndex: 4 },
      target: { seqIndex: 5, nodeIndex: 4 },
      value: 0.6,
    },

    // From convolved q, k (seq 4, 5) to elementwise product (seq 6) - 1:1 mapping
    {
      source: { seqIndex: 4, nodeIndex: 0 },
      target: { seqIndex: 6, nodeIndex: 0 },
      value: 0.5,
    },
    {
      source: { seqIndex: 5, nodeIndex: 0 },
      target: { seqIndex: 6, nodeIndex: 0 },
      value: 0.5,
    },

    {
      source: { seqIndex: 4, nodeIndex: 1 },
      target: { seqIndex: 6, nodeIndex: 1 },
      value: 0.3,
    },
    {
      source: { seqIndex: 5, nodeIndex: 1 },
      target: { seqIndex: 6, nodeIndex: 1 },
      value: 0.7,
    },

    {
      source: { seqIndex: 4, nodeIndex: 2 },
      target: { seqIndex: 6, nodeIndex: 2 },
      value: 0.6,
    },
    {
      source: { seqIndex: 5, nodeIndex: 2 },
      target: { seqIndex: 6, nodeIndex: 2 },
      value: 0.4,
    },

    {
      source: { seqIndex: 4, nodeIndex: 3 },
      target: { seqIndex: 6, nodeIndex: 3 },
      value: 0.2,
    },
    {
      source: { seqIndex: 5, nodeIndex: 3 },
      target: { seqIndex: 6, nodeIndex: 3 },
      value: 0.8,
    },

    {
      source: { seqIndex: 4, nodeIndex: 4 },
      target: { seqIndex: 6, nodeIndex: 4 },
      value: 0.4,
    },
    {
      source: { seqIndex: 5, nodeIndex: 4 },
      target: { seqIndex: 6, nodeIndex: 4 },
      value: 0.6,
    },

    // From elementwise product (seq 6) through Hyena operator to hyena output (seq 7)
    // Each output connects to all inputs (hyena has wide receptive field)
    {
      source: { seqIndex: 6, nodeIndex: 0 },
      target: { seqIndex: 7, nodeIndex: 0 },
      value: 0.3,
    },
    {
      source: { seqIndex: 6, nodeIndex: 1 },
      target: { seqIndex: 7, nodeIndex: 0 },
      value: 0.2,
    },
    {
      source: { seqIndex: 6, nodeIndex: 2 },
      target: { seqIndex: 7, nodeIndex: 0 },
      value: 0.3,
    },
    {
      source: { seqIndex: 6, nodeIndex: 3 },
      target: { seqIndex: 7, nodeIndex: 0 },
      value: 0.1,
    },
    {
      source: { seqIndex: 6, nodeIndex: 4 },
      target: { seqIndex: 7, nodeIndex: 0 },
      value: 0.1,
    },

    {
      source: { seqIndex: 6, nodeIndex: 0 },
      target: { seqIndex: 7, nodeIndex: 1 },
      value: 0.2,
    },
    {
      source: { seqIndex: 6, nodeIndex: 1 },
      target: { seqIndex: 7, nodeIndex: 1 },
      value: 0.4,
    },
    {
      source: { seqIndex: 6, nodeIndex: 2 },
      target: { seqIndex: 7, nodeIndex: 1 },
      value: 0.3,
    },
    {
      source: { seqIndex: 6, nodeIndex: 3 },
      target: { seqIndex: 7, nodeIndex: 1 },
      value: 0.1,
    },

    {
      source: { seqIndex: 6, nodeIndex: 0 },
      target: { seqIndex: 7, nodeIndex: 2 },
      value: 0.1,
    },
    {
      source: { seqIndex: 6, nodeIndex: 1 },
      target: { seqIndex: 7, nodeIndex: 2 },
      value: 0.2,
    },
    {
      source: { seqIndex: 6, nodeIndex: 2 },
      target: { seqIndex: 7, nodeIndex: 2 },
      value: 0.4,
    },
    {
      source: { seqIndex: 6, nodeIndex: 3 },
      target: { seqIndex: 7, nodeIndex: 2 },
      value: 0.2,
    },
    {
      source: { seqIndex: 6, nodeIndex: 4 },
      target: { seqIndex: 7, nodeIndex: 2 },
      value: 0.1,
    },

    {
      source: { seqIndex: 6, nodeIndex: 1 },
      target: { seqIndex: 7, nodeIndex: 3 },
      value: 0.1,
    },
    {
      source: { seqIndex: 6, nodeIndex: 2 },
      target: { seqIndex: 7, nodeIndex: 3 },
      value: 0.2,
    },
    {
      source: { seqIndex: 6, nodeIndex: 3 },
      target: { seqIndex: 7, nodeIndex: 3 },
      value: 0.4,
    },
    {
      source: { seqIndex: 6, nodeIndex: 4 },
      target: { seqIndex: 7, nodeIndex: 3 },
      value: 0.3,
    },

    {
      source: { seqIndex: 6, nodeIndex: 2 },
      target: { seqIndex: 7, nodeIndex: 4 },
      value: 0.2,
    },
    {
      source: { seqIndex: 6, nodeIndex: 3 },
      target: { seqIndex: 7, nodeIndex: 4 },
      value: 0.3,
    },
    {
      source: { seqIndex: 6, nodeIndex: 4 },
      target: { seqIndex: 7, nodeIndex: 4 },
      value: 0.5,
    },

    // From v (seq 3) through Short Explicit Conv to convolved v (seq 8)
    {
      source: { seqIndex: 3, nodeIndex: 0 },
      target: { seqIndex: 8, nodeIndex: 0 },
      value: 0.4,
    },
    {
      source: { seqIndex: 3, nodeIndex: 1 },
      target: { seqIndex: 8, nodeIndex: 0 },
      value: 0.3,
    },
    {
      source: { seqIndex: 3, nodeIndex: 2 },
      target: { seqIndex: 8, nodeIndex: 0 },
      value: 0.3,
    },

    {
      source: { seqIndex: 3, nodeIndex: 0 },
      target: { seqIndex: 8, nodeIndex: 1 },
      value: 0.2,
    },
    {
      source: { seqIndex: 3, nodeIndex: 1 },
      target: { seqIndex: 8, nodeIndex: 1 },
      value: 0.5,
    },
    {
      source: { seqIndex: 3, nodeIndex: 2 },
      target: { seqIndex: 8, nodeIndex: 1 },
      value: 0.3,
    },

    {
      source: { seqIndex: 3, nodeIndex: 1 },
      target: { seqIndex: 8, nodeIndex: 2 },
      value: 0.3,
    },
    {
      source: { seqIndex: 3, nodeIndex: 2 },
      target: { seqIndex: 8, nodeIndex: 2 },
      value: 0.4,
    },
    {
      source: { seqIndex: 3, nodeIndex: 3 },
      target: { seqIndex: 8, nodeIndex: 2 },
      value: 0.3,
    },

    {
      source: { seqIndex: 3, nodeIndex: 2 },
      target: { seqIndex: 8, nodeIndex: 3 },
      value: 0.3,
    },
    {
      source: { seqIndex: 3, nodeIndex: 3 },
      target: { seqIndex: 8, nodeIndex: 3 },
      value: 0.4,
    },
    {
      source: { seqIndex: 3, nodeIndex: 4 },
      target: { seqIndex: 8, nodeIndex: 3 },
      value: 0.3,
    },

    {
      source: { seqIndex: 3, nodeIndex: 3 },
      target: { seqIndex: 8, nodeIndex: 4 },
      value: 0.3,
    },
    {
      source: { seqIndex: 3, nodeIndex: 2 },
      target: { seqIndex: 8, nodeIndex: 4 },
      value: 0.3,
    },
    {
      source: { seqIndex: 3, nodeIndex: 4.1 },
      target: { seqIndex: 8, nodeIndex: 4 },
      value: 0.3,
    },
    {
      source: { seqIndex: 3, nodeIndex: 4 },
      target: { seqIndex: 8, nodeIndex: 4 },
      value: 0.7,
    },

    // From hyena output (seq 7) and convolved v (seq 8) to final output (seq 9) - elementwise gate (1:1)
    {
      source: { seqIndex: 7, nodeIndex: 0 },
      target: { seqIndex: 9, nodeIndex: 0 },
      value: 0.4,
    },
    {
      source: { seqIndex: 8, nodeIndex: 0 },
      target: { seqIndex: 9, nodeIndex: 0 },
      value: 0.6,
    },

    {
      source: { seqIndex: 7, nodeIndex: 1 },
      target: { seqIndex: 9, nodeIndex: 1 },
      value: 0.3,
    },
    {
      source: { seqIndex: 8, nodeIndex: 1 },
      target: { seqIndex: 9, nodeIndex: 1 },
      value: 0.7,
    },

    {
      source: { seqIndex: 7, nodeIndex: 2 },
      target: { seqIndex: 9, nodeIndex: 2 },
      value: 0.5,
    },
    {
      source: { seqIndex: 8, nodeIndex: 2 },
      target: { seqIndex: 9, nodeIndex: 2 },
      value: 0.5,
    },

    {
      source: { seqIndex: 7, nodeIndex: 3 },
      target: { seqIndex: 9, nodeIndex: 3 },
      value: 0.2,
    },
    {
      source: { seqIndex: 8, nodeIndex: 3 },
      target: { seqIndex: 9, nodeIndex: 3 },
      value: 0.8,
    },

    {
      source: { seqIndex: 7, nodeIndex: 4 },
      target: { seqIndex: 9, nodeIndex: 4 },
      value: 0.6,
    },
    {
      source: { seqIndex: 8, nodeIndex: 4 },
      target: { seqIndex: 9, nodeIndex: 4 },
      value: 0.4,
    },
  ];

  // Scales
  const xScale = d3.scaleLinear().domain([0, maxX]).range([0, boundsWidth]);

  const yScale = d3.scaleLinear().domain([0, maxY]).range([boundsHeight, 0]);

  const allValues = dnaSeqs.flatMap(d =>
    d.histogramHeights ? d.histogramHeights : []
  );
  // find the maximum
  const maxHistValue = d3.max(allValues);
  const stops = 5;
  const domainStops = d3.range(stops + 1).map(i => (i * maxHistValue) / stops);

  const greens = [
    '#265600', // darkest
    '#3f8500',
    '#76b900',
    '#bff230',
    '#cfff40', // brightest
  ];

  const colorScale = d3
    .scaleLinear()
    .domain(domainStops) // [0, max/5, 2max/5, …, max]
    .range(greens);

  // Helper function to get node position
  const getNodePosition = (seqIndex, nodeIndex) => {
    const seq = dnaSeqs[seqIndex];
    const blockWidth = 22;
    const spacing = 6;
    const rectWidth = (blockWidth + spacing) * 5; // 5 bases
    const nodeX =
      xScale(seq.x) -
      rectWidth / 2 +
      nodeIndex * (blockWidth + spacing) +
      blockWidth / 2;
    const nodeY = yScale(seq.y);
    return { x: nodeX, y: nodeY };
  };

  // Function to determine which paths should be visible based on hover
  const getVisiblePaths = (hoveredSeqIndex, hoveredNodeIndex) => {
    if (hoveredSeqIndex === null || hoveredNodeIndex === null) return [];
    console.log('hoveredSeqIndex', hoveredSeqIndex);
    console.log('hoveredNodeIndex', hoveredNodeIndex);

    const visiblePaths = [];

    // For all sequences, trace back through the entire chain
    // but only for the specific hovered node position
    const findUpstreamPaths = (
      currentSeqIndex,
      currentNodeIndex,
      visited = new Set()
    ) => {
      const pathKey = `${currentSeqIndex}-${currentNodeIndex}`;
      if (visited.has(pathKey)) return;
      visited.add(pathKey);

      sankeyLinks.forEach((link, linkIndex) => {
        // console.log("matching link", linkIndex, link);
        // If this link leads TO our current position
        if (
          link.target.seqIndex === currentSeqIndex &&
          link.target.nodeIndex === currentNodeIndex
        ) {
          visiblePaths.push(linkIndex);

          // Recursively find paths leading to the source of this link
          findUpstreamPaths(
            link.source.seqIndex,
            link.source.nodeIndex,
            visited
          );
        }
      });
    };

    // Start the recursive search from the hovered position
    findUpstreamPaths(hoveredSeqIndex, hoveredNodeIndex);
    console.log('visiblePaths', visiblePaths);
    return visiblePaths;
  };
  const createSankeyPath = link => {
    const sourcePos = getNodePosition(
      link.source.seqIndex,
      link.source.nodeIndex
    );
    const targetPos = getNodePosition(
      link.target.seqIndex,
      link.target.nodeIndex
    );

    // Check if the path is vertical (same x-coordinate)
    const isVertical = Math.abs(sourcePos.x - targetPos.x) < 0.001; // Small threshold for floating-point precision
    console.log('isVertical', isVertical, sourcePos.x, targetPos.x);
    if (isVertical) {
      // For vertical paths, draw a straight line
      return `M ${sourcePos.x} ${sourcePos.y} L ${targetPos.x} ${targetPos.y}`;
    } else {
      // For non-vertical paths, use the original Bézier curve
      const midY = (sourcePos.y + targetPos.y) / 2;
      const controlPoint1 = { x: sourcePos.x, y: midY };
      const controlPoint2 = { x: targetPos.x, y: midY };

      return `M ${sourcePos.x} ${sourcePos.y} 
              C ${controlPoint1.x} ${controlPoint1.y}, 
                ${controlPoint2.x} ${controlPoint2.y}, 
                ${targetPos.x} ${targetPos.y}`;
    }
  };

  // Grid lines
  const yGridLines = () => {
    return yScale
      .ticks()
      .map(i => (
        <line
          key={i}
          x1={xScale(1.5)}
          x2={xScale(maxX)}
          y1={yScale(i)}
          y2={yScale(i)}
          stroke={strokeColor}
          strokeDasharray="2,2"
          strokeWidth={strokeWidth}
          opacity={[1, 3, 5, 7].includes(i) ? gridOpacity : 0}
        />
      ));
  };

  // Components
  const DNASequence = ({
    x,
    y,
    label = null,
    histogramHeights = null,
    verticalOffset = 0,
  }) => {
    const blockWidth = 22;
    const blockHeight = 20;
    const spacing = 6;
    const sequence = 'ATCGA';
    const rectWidth = (blockWidth + spacing) * sequence.length;
    const rectHeight = blockHeight + spacing * 2;
    const outerRectSpacing = 2;
    const maxHistogramHeight = 15;

    return (
      <g
        transform={`translate(${xScale(x) - rectWidth / 2 - outerRectSpacing}, ${yScale(y) - rectHeight / 2})`}
      >
        <text
          x={spacing}
          y={-4}
          fontSize="10px"
          fontWeight="100"
          fontFamily="NVIDIA Sans, -apple-system, sans-serif"
          fill="black"
          textAnchor="middle"
        >
          {label}
        </text>
        <rect
          width={rectWidth + outerRectSpacing * 2}
          height={rectHeight}
          fill="white"
          stroke={strokeColor}
          rx={4}
          strokeWidth={1}
        />
        {sequence.split('').map((base, i) => {
          const weight = histogramHeights ? histogramHeights[i] : 0;
          const histogramHeight = weight * maxHistogramHeight;
          const isHovered =
            hoveredIndex === i && verticalOffsetIndex >= verticalOffset;
          const isDimmed =
            (hoveredIndex !== null && hoveredIndex !== i) ||
            (verticalOffsetIndex !== null &&
              verticalOffsetIndex < verticalOffset);
          //  && verticalOffsetIndex >= verticalOffset;

          return (
            <g key={i} transform={`translate(${spacing / 2}, ${spacing})`}>
              {/* Histogram bar */}
              {histogramHeights && (
                <rect
                  x={
                    i * (blockWidth + spacing) +
                    outerRectSpacing +
                    blockWidth / 4
                  }
                  y={-histogramHeight - spacing - (isHovered ? 2 : 0)}
                  width={blockWidth / 2}
                  height={histogramHeight + (isHovered ? 2 : 0)}
                  fill="#334155"
                  opacity={isDimmed ? 0.3 : 0.7}
                  rx={1}
                  style={{ transition: 'all 0.2s ease' }}
                />
              )}
              <rect
                x={i * (blockWidth + spacing) + outerRectSpacing}
                width={blockWidth}
                height={blockHeight}
                fill={histogramHeights ? colorScale(weight) : 'var(--silver)'}
                stroke={strokeColor}
                rx={3}
                strokeWidth={0}
                opacity={isDimmed ? 0.4 : 1}
                onMouseEnter={() => {
                  setHoveredIndex(i);
                  setVerticalOffsetIndex(verticalOffset);
                  setHoveredSeqIndex(
                    dnaSeqs.findIndex(seq => seq.x === x && seq.y === y)
                  );
                }}
                onMouseLeave={() => {
                  setHoveredIndex(null);
                  setVerticalOffsetIndex(null);
                  setHoveredSeqIndex(null);
                }}
                style={{ cursor: 'pointer', transition: 'opacity 0.2s ease' }}
              />
              <text
                x={
                  i * (blockWidth + spacing) + blockWidth / 2 + outerRectSpacing
                }
                y={blockHeight / 2 + 4}
                textAnchor="middle"
                fontSize="10px"
                fontWeight="500"
                fontFamily="NVIDIA Sans, -apple-system, sans-serif"
                fill={
                  histogramHeights && weight < domainStops[1]
                    ? 'white'
                    : 'black'
                }
                opacity={isDimmed ? 0.4 : 1}
                style={{
                  textShadow: '0 1px 1px rgba(0,0,0,0.3)',
                  pointerEvents: 'none',
                  transition: 'opacity 0.2s ease',
                }}
              >
                {base}
              </text>
              {/* Weight tooltip on hover */}
              {isHovered && histogramHeights && (
                <g
                  transform={`translate(${i * (blockWidth + spacing) + blockWidth / 2 + outerRectSpacing}, -10)`}
                >
                  <rect
                    x={-15}
                    y={-18}
                    width={30}
                    height={16}
                    fill="black"
                    opacity={0.8}
                    rx={3}
                  />
                  <text
                    x={0}
                    y={-6}
                    textAnchor="middle"
                    fontSize="9px"
                    fontWeight="600"
                    fill="white"
                  >
                    {weight.toFixed(2)}
                  </text>
                </g>
              )}
              {i < sequence.length - 1 && (
                <line
                  x1={
                    i * (blockWidth + spacing) +
                    blockWidth +
                    outerRectSpacing +
                    spacing / 2
                  }
                  y1={-spacing}
                  x2={
                    i * (blockWidth + spacing) +
                    blockWidth +
                    outerRectSpacing +
                    spacing / 2
                  }
                  y2={blockHeight + spacing}
                  stroke={strokeColor}
                  strokeWidth={1}
                  opacity={isDimmed ? 0.3 : 1}
                  style={{ transition: 'opacity 0.2s ease' }}
                />
              )}
            </g>
          );
        })}
      </g>
    );
  };

  const drawArrowLine = ({ x1, y1, x2, y2, arrowSize = 4 }) => {
    const path = d3.path();
    const scaledX1 = xScale(x1);
    const scaledY1 = yScale(y1);
    const scaledX2 = xScale(x2);
    const scaledY2 = yScale(y2);

    path.moveTo(scaledX1, scaledY1);
    path.lineTo(scaledX2, scaledY2);

    // Arrow
    const angle = Math.atan2(scaledY2 - scaledY1, scaledX2 - scaledX1);
    const arrowPoint1X = scaledX2 - arrowSize * Math.cos(angle - Math.PI / 6);
    const arrowPoint1Y = scaledY2 - arrowSize * Math.sin(angle - Math.PI / 6);
    const arrowPoint2X = scaledX2 - arrowSize * Math.cos(angle + Math.PI / 6);
    const arrowPoint2Y = scaledY2 - arrowSize * Math.sin(angle + Math.PI / 6);

    path.moveTo(scaledX2, scaledY2);
    path.lineTo(arrowPoint1X, arrowPoint1Y);
    path.moveTo(scaledX2, scaledY2);
    path.lineTo(arrowPoint2X, arrowPoint2Y);

    return (
      <path
        d={path.toString()}
        fill="none"
        stroke={strokeColor}
        strokeWidth={hoveredIndex === null ? strokeWidth : 0}
        style={{ transition: 'all 0.3s ease' }}
      />
    );
  };

  const bezierConnectPath = ({
    x1,
    y1,
    x2,
    y2,
    controlOffset = 0,
    bendFactor = 0.8,
    ...pathProps
  }) => {
    const path = d3.path();
    const startX = xScale(x1);
    const startY = yScale(y1);
    const endX = xScale(x2);
    const endY = yScale(y2);
    const dx = endX - startX;
    const dy = endY - startY;
    const isHorizontal = Math.abs(dx) > Math.abs(dy);

    let cp1x, cp1y, cp2x, cp2y;

    if (isHorizontal) {
      cp1x = startX + dx * controlOffset;
      cp1y = startY + dy * bendFactor;
      cp2x = endX - dx * controlOffset;
      cp2y = endY - dy * bendFactor;
    } else {
      cp1x = startX + dx * bendFactor;
      cp1y = startY + dy * controlOffset;
      cp2x = endX - dx * bendFactor;
      cp2y = endY - dy * controlOffset;
    }

    path.moveTo(startX, startY);
    path.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, endX, endY);

    return (
      <path
        d={path.toString()}
        fill="none"
        stroke={strokeColor}
        strokeWidth={hoveredIndex === null ? strokeWidth : 0}
        style={{ transition: 'all 0.3s ease' }}
        {...pathProps}
      />
    );
  };

  const drawOperation = (x, y, text = null) => {
    const width = 140;
    const height = 20;
    return (
      <g
        transform={`translate(${xScale(x) - width / 2}, ${yScale(y) - height / 2})`}
      >
        <rect
          width={width}
          height={height}
          fill="white"
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          rx={8}
        />
        <text
          x={width / 2}
          y={height / 2 + 3}
          textAnchor="middle"
          fontSize="10px"
          fontWeight="500"
          fontFamily="NVIDIA Sans, -apple-system, sans-serif"
          fill="black"
        >
          {text || ''}
        </text>
      </g>
    );
  };

  const drawText = (x, y, text, chunkSize = 6) => {
    const lineHeight = 18;

    // Split text into chunks
    const textChunks = [];
    const words = text.split(' ');
    for (let i = 0; i < words.length; i += chunkSize) {
      textChunks.push(words.slice(i, i + chunkSize).join(' '));
    }

    // Calculate vertical offset
    const totalTextHeight = textChunks.length * lineHeight;
    const verticalOffset = totalTextHeight / 2;

    return (
      <g
        transform={`translate(${xScale(x)}, ${yScale(y) - verticalOffset / 2})`}
      >
        <text
          x={0}
          y={0}
          fontSize="10px"
          fontWeight="500"
          fontFamily="NVIDIA Sans, -apple-system, sans-serif"
          fill="black"
          textAnchor="start"
        >
          {textChunks.map((chunk, index) => (
            <tspan key={index} x={0} y={index * lineHeight}>
              {chunk}
            </tspan>
          ))}
        </text>
      </g>
    );
  };

  return (
    <div style={{ margin: '0 auto', width: '900px' }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{
          fontFamily:
            'NVIDIA Sans, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        }}
      >
        <defs>
          {/* Gradient for Sankey paths */}
          <linearGradient id="sankeyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="var(--nvgreen)" stopOpacity="0.4" />
            <stop offset="100%" stopColor="var(--nvgreen)" stopOpacity="0.2" />
          </linearGradient>

          {/* Glow filter for paths */}
          <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <g
          width={boundsWidth}
          height={boundsHeight}
          transform={`translate(${MARGIN.left}, ${MARGIN.top})`}
          overflow="visible"
        >
          {/* Grid lines */}
          {yGridLines()}

          {/* Arrow lines */}
          {arrowLines.map((props, i) => (
            <React.Fragment key={`arrow-${i}`}>
              {drawArrowLine(props)}
            </React.Fragment>
          ))}

          {/* Curved arrow lines */}
          {curvedArrowLines.map((props, i) => (
            <React.Fragment key={`curved-arrow-${i}`}>
              {bezierConnectPath(props)}
            </React.Fragment>
          ))}

          {/* Sankey paths - only show on hover */}
          {hoveredSeqIndex !== null && hoveredIndex !== null && (
            <g>
              {getVisiblePaths(hoveredSeqIndex, hoveredIndex).map(linkIndex => {
                const link = sankeyLinks[linkIndex];
                return (
                  <path
                    key={`sankey-${linkIndex}`}
                    d={createSankeyPath(link)}
                    fill="none"
                    // stroke="url(#sankeyGradient)"
                    stroke="var(--silver)"
                    // strokeWidth={Math.max(2, link.value * 6)}
                    strokeWidth={8}
                    opacity={0.75}
                    // filter="url(#glow)"
                    style={{
                      transition: 'all 0.3s ease',
                      pointerEvents: 'none',
                    }}
                  />
                );
              })}
            </g>
          )}

          {/* DNA sequences */}
          {dnaSeqs.map(({ x, y, label, histogramHeights }, i) => (
            <React.Fragment key={`dna-${i}`}>
              {DNASequence({
                x,
                y,
                label,
                histogramHeights,
                verticalOffset: i,
              })}
            </React.Fragment>
          ))}

          {/* Operations */}
          {operations.map(({ x, y, label }, i) => (
            <React.Fragment key={`op-${i}`}>
              {drawOperation(x, y, label)}
            </React.Fragment>
          ))}

          {/* Text annotations */}
          {textAnnotations.map(({ x, y, text, chunkSize }, i) => (
            <React.Fragment key={`text-${i}`}>
              {drawText(x, y, text, chunkSize)}
            </React.Fragment>
          ))}
        </g>
      </svg>
    </div>
  );
};

export default HyenaOperatorDiagram;
