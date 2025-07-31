import React, { useMemo } from 'react';
import * as d3 from 'd3';

const SparkDesignBar = ({
  dnaSequence = '',
  peakPattern = [],
  height = 18,
  nodeSize = 5,
  barColor = 'var(--nvgreen)',
  peakColor = 'black',
  strokeColor = 'currentColor', // color for x-axis line
  strokeWidth = 1,
}) => {
  const dnaValueMap = { A: 2, T: 4, C: 6, G: 8 };

  // numeric bar heights (0–10)
  const data = useMemo(
    () =>
      dnaSequence
        .toUpperCase()
        .split('')
        .map(b => dnaValueMap[b] ?? 0),
    [dnaSequence]
  );

  // size chart to the longer of dna vs peaks
  const barCount = Math.max(data.length, peakPattern.length);
  if (barCount === 0) return null;

  const width = barCount * nodeSize;
  const boundsHeight = height;

  // scales
  const xScale = useMemo(
    () =>
      d3.scaleBand().domain(d3.range(barCount)).range([0, width]).padding(0),
    [barCount, width]
  );
  const yScale = useMemo(
    () => d3.scaleLinear().domain([0, 8]).range([boundsHeight, 0]),
    [boundsHeight]
  );

  // build a path of stroked boxes where peakPattern[i]===1
  const peakPath = useMemo(() => {
    if (!peakPattern.length) return null;
    let path = '';
    let inPeak = false;
    let startX = 0;

    for (let i = 0; i < peakPattern.length; i++) {
      if (peakPattern[i] === 1 && !inPeak) {
        startX = xScale(i);
        inPeak = true;
      } else if (peakPattern[i] === 0 && inPeak) {
        const endX = xScale(i - 1) + xScale.bandwidth();
        path += `M${startX},0 H${endX} V${boundsHeight} H${startX} Z `;
        inPeak = false;
      }
    }
    if (inPeak) {
      const last = peakPattern.length - 1;
      const endX = xScale(last) + xScale.bandwidth();
      path += `M${startX},0 H${endX} V${boundsHeight} H${startX} Z `;
    }
    return path;
  }, [peakPattern, xScale, boundsHeight]);

  return (
    <svg
      width={width}
      height={height}
      style={{
        display: 'inline-block',
        verticalAlign: 'text-bottom',
        overflow: 'visible',
        marginBottom: '0.25rem',
      }}
    >
      {/* DNA bars */}
      {data.map((val, i) =>
        i < barCount ? (
          <rect
            key={`dna-${i}`}
            x={xScale(i)}
            y={yScale(val)}
            width={xScale.bandwidth()}
            height={boundsHeight - yScale(val)}
            fill={barColor}
          />
        ) : null
      )}

      {/* peak rectangles (stroke only) */}
      {peakPath && (
        <path
          d={peakPath}
          fill="transparent"
          stroke={peakColor}
          strokeWidth={strokeWidth}
        />
      )}

      {/* baseline / x-axis line with strokeColor */}
      <line
        x1={0}
        x2={width}
        y1={boundsHeight}
        y2={boundsHeight}
        stroke={strokeColor}
        strokeWidth={strokeWidth}
      />
    </svg>
  );
};

export default SparkDesignBar;
