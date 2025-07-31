import React, { useMemo } from 'react';
import * as d3 from 'd3';

const CONFIG = {
  STYLES: {
    STROKE_COLOR: '#333',
    STROKE_WIDTH: 1,
    ANNOTATION_STROKE_WIDTH: 2,
  },
  BASE_COLORS: {
    A: { fill: '#ff6b6b', stroke: '#c92a2a' },
    T: { fill: 'var(--inch-worm)', stroke: 'var(--nvgreen)' },
    C: { fill: 'var(--pippin)', stroke: 'var(--geraldine)' },
    G: { fill: '#95e1d3', stroke: '#20c997' },
  },
};

const AnnotatedDnaGrid = ({
  cellSize = 17, // uniform height and width
  cellCount = 35,
  margin = { top: 0, bottom: 5, left: 10, right: 30 },
  annotationCellIndex = null,
  annotationText = 'Perturbing this could have a large effect!',
  cellReplacement = 'T',
}) => {
  // Add margin-left and margin-right calculations
  const marginLeft = margin.left !== undefined ? margin.left : 30;
  const marginRight = margin.right !== undefined ? margin.right : 30;

  const actualWidth = cellCount * cellSize;
  const totalWidth = marginLeft + actualWidth + marginRight;
  const totalHeight =
    (margin.top !== undefined ? margin.top : 60) +
    cellSize +
    (margin.bottom !== undefined ? margin.bottom : 40);

  // simple identity scales mapping "pixel → pixel"
  const xScale = d3
    .scaleLinear()
    .domain([0, totalWidth])
    .range([0, totalWidth]);
  const yScale = d3
    .scaleLinear()
    .domain([0, totalHeight])
    .range([0, totalHeight]);

  // generate a random A/T/C/G sequence once
  // Use a fixed 30-base DNA sequence string instead of random generation
  const sequence = useMemo(() => {
    // Example: 30 bases, can be any valid DNA sequence
    return 'ATGCTAGCTAGGCTAACGTTAGCTAGCTAG'.split('');
  }, []);

  // Bezier connection function
  const BezierConnection = ({
    x1,
    y1,
    x2,
    y2,
    controlOffset = 0,
    bendFactor = 0.8,
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
        stroke={CONFIG.STYLES.STROKE_COLOR}
        strokeWidth={CONFIG.STYLES.STROKE_WIDTH}
        strokeDasharray="4,4"
      />
    );
  };

  return (
    <svg width={totalWidth} height={totalHeight}>
      {/* Annotation arrow and text pointing to the annotated cell */}
      {annotationCellIndex !== null && (
        <>
          <BezierConnection
            x1={marginLeft + cellSize * annotationCellIndex}
            y1={margin.top - 30}
            x2={marginLeft + cellSize * annotationCellIndex + cellSize / 2}
            y2={margin.top + cellSize / 2}
            controlOffset={0.2}
            bendFactor={-1}
          />
          {annotationText && (
            <text
              x={marginLeft + cellSize * annotationCellIndex}
              y={margin.top - 35}
              textAnchor="middle"
              fontFamily="Nvidia Sans"
              fontSize={9.5}
              fill="var(--mine-shaft)"
              strokeWidth="1"
              fontWeight="bold"
            >
              {annotationText}
            </text>
          )}
        </>
      )}
      {/* DNA grid */}
      {sequence.map((base, i) => {
        const isAnnotated = annotationCellIndex === i;
        // If this is the annotated cell, show the replacement base and color
        const displayBase = isAnnotated ? cellReplacement : base;
        const baseColors = CONFIG.BASE_COLORS[displayBase] || {
          fill: 'none',
          stroke: '#ccc',
        };

        return (
          <g key={i}>
            <rect
              x={marginLeft + i * cellSize}
              y={margin.top}
              width={cellSize}
              height={cellSize}
              fill={isAnnotated ? baseColors.fill : 'none'}
              stroke="#ccc"
              strokeWidth={1}
            />
            <text
              x={marginLeft + i * cellSize + cellSize / 2}
              y={margin.top + cellSize / 2 + 3}
              textAnchor="middle"
              fontFamily="Nvidia Sans"
              fontSize={cellSize * 0.6}
              fill="black"
              strokeWidth="1"
              fontWeight={isAnnotated ? 'bold' : 'normal'}
            >
              {displayBase}
            </text>
          </g>
        );
      })}
      {/* Black stroke rectangle for annotated elements, moved out of the main loop */}
      {annotationCellIndex !== null && (
        <rect
          x={marginLeft + annotationCellIndex * cellSize}
          y={margin.top}
          width={cellSize}
          height={cellSize}
          fill="none"
          stroke={CONFIG.BASE_COLORS[cellReplacement].stroke}
          strokeWidth={CONFIG.STYLES.ANNOTATION_STROKE_WIDTH}
        />
      )}
    </svg>
  );
};

export default AnnotatedDnaGrid;
