import React, { useMemo } from 'react';
import * as d3 from 'd3';

const DesignBar = ({
  dnaSequence = 'ATCGCGATTACGAT', // Default sequence
  borzoi = [],
  enformer = [],
  peakPattern = [],
  nodeSize = 14, // ← matches your DNASequence nodeSize
  height = 200,
  barColor = 'var(--nvgreen)',
  borzoiColor = '#FF6347',
  enformerColor = '#32CD32',
  strokeWidth = 1,
  curveType = d3.curveBasis,
}) => {
  // map bases to values
  const dnaValueMap = { A: 2, T: 4, C: 6, G: 8 };

  // build data array from dnaSequence
  const data = useMemo(
    () =>
      dnaSequence
        .toUpperCase()
        .split('')
        .map(b => dnaValueMap[b] || 0),
    [dnaSequence]
  );

  const MARGIN = { top: 10, right: 0, bottom: 20, left: 0 };
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;
  const barCount = data.length;

  // total width needed so that each bar is exactly nodeSize px
  const boundsWidth = nodeSize * barCount;
  const svgWidth = boundsWidth + MARGIN.left + MARGIN.right;

  // scales: no padding, so bandwidth() === nodeSize
  const xScale = useMemo(
    () =>
      d3
        .scaleBand()
        .domain(d3.range(barCount))
        .range([0, boundsWidth])
        .padding(0),
    [barCount, boundsWidth]
  );

  const yScale = useMemo(
    () => d3.scaleLinear().domain([0, 10]).range([boundsHeight, 0]),
    [boundsHeight]
  );

  // peak‐pattern path (unchanged)
  const peakPatternPath = useMemo(() => {
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
        path += `M${startX},${yScale(10)} H${endX} V${yScale(0)} H${startX} Z `;
        inPeak = false;
      }
    }
    if (inPeak) {
      const last = peakPattern.length - 1;
      const endX = xScale(last) + xScale.bandwidth();
      path += `M${startX},${yScale(10)} H${endX} V${yScale(0)} H${startX} Z `;
    }
    return path;
  }, [peakPattern, xScale, yScale]);

  // density paths (unchanged)
  const densityPaths = useMemo(() => {
    const makePath = arr =>
      arr.length
        ? d3
            .line()
            .x((_, i) => xScale(i) + xScale.bandwidth() / 2)
            .y(d => yScale(d))
            .curve(curveType)(arr)
        : null;
    return {
      borzoiPath: makePath(borzoi),
      enformerPath: makePath(enformer),
    };
  }, [borzoi, enformer, xScale, yScale, curveType]);

  if (!data.length) return null;

  return (
    <svg width={svgWidth} height={height}>
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        {/* Bars now exactly nodeSize wide */}
        {data.map((val, i) => (
          <rect
            key={i}
            x={xScale(i)}
            y={yScale(val)}
            width={xScale.bandwidth()}
            height={boundsHeight - yScale(val)}
            fill={barColor}
            opacity={0.9}
            rx={1}
            ry={1}
          />
        ))}

        {/* Borzoi path */}
        {densityPaths.borzoiPath && (
          <path
            d={densityPaths.borzoiPath}
            stroke={borzoiColor}
            strokeWidth={2}
            fill="none"
          />
        )}

        {/* Enformer path */}
        {densityPaths.enformerPath && (
          <path
            d={densityPaths.enformerPath}
            stroke={enformerColor}
            strokeWidth={2}
            fill="none"
          />
        )}

        {/* Peak pattern */}
        {peakPatternPath && (
          <path
            d={peakPatternPath}
            stroke="black"
            strokeWidth={strokeWidth}
            fill="transparent"
          />
        )}

        {/* X axis line */}
        <line
          x1={0}
          x2={boundsWidth}
          y1={boundsHeight}
          y2={boundsHeight}
          stroke="#000"
          strokeWidth={strokeWidth}
        />
      </g>
    </svg>
  );
};

export default DesignBar;
