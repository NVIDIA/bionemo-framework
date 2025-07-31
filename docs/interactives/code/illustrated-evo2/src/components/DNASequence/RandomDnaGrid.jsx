// RandomDnaGrid.jsx
import React, { useMemo } from 'react';

const RandomDnaGrid = ({
  width = 800,
  height = 40,
  cells = 40,
  fill = '#def', // background fill for each cell
  stroke = '#ccc', // border color for each cell
  fontFamily = 'monospace',
}) => {
  // generate one random sequence per render
  const sequence = useMemo(() => {
    const bases = ['A', 'T', 'C', 'G'];
    return Array.from(
      { length: cells },
      () => bases[Math.floor(Math.random() * 4)]
    );
  }, [cells]);

  const cellWidth = width / cells;
  const fontSize = Math.min(cellWidth * 0.6, height * 0.8);

  return (
    <svg width={width} height={height}>
      {sequence.map((base, i) => {
        const x = i * cellWidth;
        return (
          <g key={i}>
            <rect
              x={x}
              y={0}
              width={cellWidth}
              height={height}
              fill={fill}
              stroke={stroke}
            />
            <text
              x={x + cellWidth / 2}
              y={height / 2 + fontSize / 3}
              textAnchor="middle"
              fontFamily={fontFamily}
              fontSize={fontSize}
              fill="#000"
            >
              {base}
            </text>
          </g>
        );
      })}
    </svg>
  );
};

export default RandomDnaGrid;
