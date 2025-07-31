import * as d3 from 'd3';

const CONFIG = {
  STYLES: {
    STROKE_COLOR: '#333',
    STROKE_WIDTH: 1,
    ANNOTATION_STROKE_WIDTH: 2,
  },
};

const SNVWindow = ({
  cellSize = 17,
  margin = { top: 60, bottom: 40, left: 10, right: 10 },
  annotationText = 'Perturbing this could have a large effect!',
}) => {
  // Define the three sequence blocks
  const leftSequence = 'ATC';
  const middleSequence = 'ATGCGEAGTGCAT';
  const rightSequence = 'TEC';
  const ellipsisWidth = cellSize * 2; // Width for "..." text

  // Calculate positions
  const leftBlockWidth = leftSequence.length * cellSize;
  const middleBlockWidth = middleSequence.length * cellSize;
  const rightBlockWidth = rightSequence.length * cellSize;

  const marginLeft = margin.left;
  const marginRight = margin.right;

  const spaceCollapseBetweenBlocks = 7;

  // Calculate x positions for each block
  const leftBlockX = marginLeft;
  const leftEllipsisX =
    leftBlockX + leftBlockWidth - spaceCollapseBetweenBlocks;
  const middleBlockX =
    leftEllipsisX + ellipsisWidth - spaceCollapseBetweenBlocks;
  const rightEllipsisX =
    middleBlockX + middleBlockWidth - spaceCollapseBetweenBlocks;
  const rightBlockX =
    rightEllipsisX + ellipsisWidth - spaceCollapseBetweenBlocks;

  const totalWidth = rightBlockX + rightBlockWidth + marginRight;
  const totalHeight = margin.top + cellSize + margin.bottom;

  // Find the middle element in the middle sequence
  const middleElementIndex = Math.floor(middleSequence.length / 2);
  const annotationCellX = middleBlockX + middleElementIndex * cellSize;

  // scales
  const xScale = d3
    .scaleLinear()
    .domain([0, totalWidth])
    .range([0, totalWidth]);
  const yScale = d3
    .scaleLinear()
    .domain([0, totalHeight])
    .range([0, totalHeight]);

  // Arc connection function using d3.arc
  const ArcConnection = ({ centerX, centerY, targetX, targetY, radius }) => {
    // Calculate the angle from center to target
    const angle = Math.atan2(targetY - centerY, targetX - centerX);

    // Create arc generator
    const arcGenerator = d3
      .arc()
      .innerRadius(radius - 1)
      .outerRadius(radius + 1)
      .startAngle(angle - 0.1)
      .endAngle(angle + 0.1);

    // Create a path from center to target
    const path = d3.path();
    path.moveTo(centerX, centerY);

    // Calculate control points for a smooth curve
    const midX = (centerX + targetX) / 2;
    const midY = Math.min(centerY, targetY) - 30;

    path.quadraticCurveTo(midX, midY, targetX, targetY);

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

  // Function to render a sequence block
  const renderSequenceBlock = (
    sequence,
    startX,
    startY,
    isMiddleBlock = false
  ) => {
    return sequence.split('').map((base, i) => {
      const x = startX + i * cellSize;
      const isMiddleElement = isMiddleBlock && i === middleElementIndex;
      const fillColor = isMiddleElement ? 'var(--inch-worm)' : 'white';

      return (
        <g key={`${startX}-${i}`}>
          <rect
            x={x}
            y={startY}
            width={cellSize}
            height={cellSize}
            fill={fillColor}
            stroke="black"
            strokeWidth={1}
          />
          <text
            x={x + cellSize / 2}
            y={startY + cellSize / 2 + 3}
            textAnchor="middle"
            fontFamily="Nvidia Sans"
            fontSize={cellSize * 0.6}
            fill="black"
            fontWeight={isMiddleElement ? 'bold' : 'normal'}
          >
            {base}
          </text>
        </g>
      );
    });
  };

  // Render ellipsis
  const renderEllipsis = (x, y) => (
    <text
      x={x + ellipsisWidth / 2}
      y={y + cellSize / 2 + 3}
      textAnchor="middle"
      fontFamily="monospace"
      fontSize={cellSize * 0.8}
      fill="black"
      fontWeight="bold"
    >
      ...
    </text>
  );

  return (
    <svg width={totalWidth} height={totalHeight}>
      {/* Annotation text */}
      <text
        x={annotationCellX + cellSize / 2}
        y={margin.top - 35}
        textAnchor="middle"
        fontFamily="Arial"
        fontSize={11}
        fill="black"
        fontWeight="bold"
      >
        {annotationText}
      </text>

      {/* Arc paths from middle element to elements adjacent to ellipses */}
      <ArcConnection
        centerX={annotationCellX + cellSize / 2}
        centerY={margin.top - cellSize * 0.1}
        targetX={middleBlockX + cellSize - 10}
        targetY={margin.top + cellSize * 0.05}
        radius={100}
      />
      <ArcConnection
        centerX={annotationCellX + cellSize / 2}
        centerY={margin.top - cellSize * 0.1}
        targetX={
          middleBlockX +
          (middleSequence.length - 1) * cellSize +
          cellSize / 2 +
          5
        }
        targetY={margin.top + cellSize * 0.05}
        radius={100}
      />

      {/* DNA sequence blocks */}
      {renderSequenceBlock(leftSequence, leftBlockX, margin.top)}
      {renderEllipsis(leftEllipsisX, margin.top)}
      {renderSequenceBlock(middleSequence, middleBlockX, margin.top, true)}
      {renderEllipsis(rightEllipsisX, margin.top)}
      {renderSequenceBlock(rightSequence, rightBlockX, margin.top)}

      {/* Index labels
      <text
        x={leftBlockX + cellSize / 2}
        y={margin.top + cellSize + 15}
        textAnchor="middle"
        fontFamily="Arial"
        fontSize={10}
        fill="black"
      >
        {startIndex}
      </text>
      <text
        x={rightBlockX + rightBlockWidth - cellSize / 2}
        y={margin.top + cellSize + 15}
        textAnchor="middle"
        fontFamily="Arial"
        fontSize={10}
        fill="black"
      >
        {endIndex}
      </text> */}
    </svg>
  );
};

export default SNVWindow;
