// ToeplitzMatrix.jsx
import React, { useRef } from 'react';
import * as d3 from 'd3';

const ToeplitzMatrix = ({ size = 'medium', label = '' }) => {
  const svgRef = useRef(null);

  // — Dimensions
  const matrixSize = 75;
  const MARGIN = { top: label ? 20 : 10, right: 10, bottom: 10, left: 10 };
  const width = matrixSize + MARGIN.left + MARGIN.right;
  const height = matrixSize + MARGIN.top + MARGIN.bottom;

  // — Matrix configuration (number of cells per side)
  const cellsPerSide = size === 'short' ? 10 : size === 'medium' ? 25 : 25;
  const greens = [
    'var(--gray-200)',
    '#2ECC71',
    '#27AE60',
    '#229954',
    '#1E8449',
    '#196F3D',
  ];

  // — Choose a "window" (number of diagonals) by type:
  //    short  → 4
  //    medium → 7
  //    long   → truly global (Infinity)
  const rawWindow = size === 'short' ? 4 : size === 'medium' ? 10 : Infinity;

  // — Clamp to at most (cellsPerSide - 1), so we never exceed the matrix
  const windowSize =
    rawWindow === Infinity ? Infinity : Math.min(rawWindow, cellsPerSide - 1);

  // — Scales
  const xScale = d3
    .scaleBand()
    .domain(d3.range(cellsPerSide))
    .range([0, matrixSize]);
  const yScale = d3
    .scaleBand()
    .domain(d3.range(cellsPerSide))
    .range([0, matrixSize]);

  // — Color scheme
  const colorScheme = d3.scaleQuantize().domain([0, 1]).range(greens);

  // — Generate all <rect> cells
  const cells = d3.range(cellsPerSide).flatMap(row =>
    d3.range(cellsPerSide).map(col => {
      const dist = Math.abs(row - col);
      let intensity;

      if (windowSize === Infinity) {
        // long: uniform global
        intensity = 1;
      } else if (dist >= windowSize) {
        // outside the window band, zero out
        intensity = 0;
      } else {
        // linear fall‐off within the window
        intensity = Math.max(0, 1 - dist / (windowSize - 1));
      }

      return (
        <rect
          key={`${row}-${col}`}
          x={xScale(col)}
          y={yScale(row)}
          width={xScale.bandwidth()}
          height={yScale.bandwidth()}
          fill={colorScheme(intensity)}
          stroke="white"
        />
      );
    })
  );

  return (
    <div style={{ margin: '0 auto', width }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{
          fontFamily:
            'NVIDIA Sans, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        }}
      >
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* Background box */}
          <rect
            width={matrixSize}
            height={matrixSize}
            fill="#fff9f9"
            stroke="#ccc"
            strokeWidth={1}
          />

          {/* Toeplitz cells */}
          {cells}

          {/* Optional label above */}
          {label && (
            <text
              x={matrixSize / 2}
              y={-5}
              textAnchor="middle"
              fontSize="8px"
              fontWeight="600"
              fill="#333"
            >
              {label}
            </text>
          )}
        </g>
      </svg>
    </div>
  );
};

export default ToeplitzMatrix;
