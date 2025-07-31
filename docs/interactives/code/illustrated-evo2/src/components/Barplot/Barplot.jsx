import React, { useMemo } from 'react';
import * as d3 from 'd3';

const MARGIN = { top: 15, right: 55, bottom: 30, left: 0 };
const BAR_PADDING = 0.2;

// Color palette using CSS variables
const FILL_COLORS = [
  'var(--hot-cinnamon)', // #ef9100
  'var(--carrot-orange)', // #ef9100
  'var(--amber)', // #f9c500
  'var(--pale-canary)', // #feeeb2
];

const STROKE_COLORS = [
  'var(--deep-fir)', // #265600
  'var(--pigment-indigo)', // #4d1368
  'var(--hot-cinnamon)', // #df6500
  'var(--tarawera)', // #002781
  'var(--dark-burgundy)', // #650b0b
  'var(--mulberry-wood)', // #5d1337
  'var(--evening-sea)', // #04554b
  'var(--limeade)', // #3f8500
];

const Barplot = ({ width, height, data }) => {
  const boundsWidth = width - MARGIN.left - MARGIN.right;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  // Sort data descending
  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => b.value - a.value);
  }, [data]);

  const maxValue = sortedData.length ? sortedData[0].value : 0;

  // X scale: groups
  const xScale = useMemo(() => {
    return d3
      .scaleBand()
      .domain(sortedData.map(d => d.group))
      .range([0, boundsWidth])
      .padding(BAR_PADDING);
  }, [sortedData, boundsWidth]);

  // Y scale: values
  const yScale = useMemo(() => {
    return d3.scaleLinear().domain([0, maxValue]).range([boundsHeight, 0]);
  }, [maxValue, boundsHeight]);

  // Color scales
  const fillColorScale = d3
    .scaleOrdinal()
    .domain(sortedData.map(d => d.group))
    .range(FILL_COLORS);

  return (
    <div className="barplot-container">
      <p className="loss-chart-title">NCBI Genomes</p>
      <svg width={width} height={height}>
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* Bars */}
          <g>
            {sortedData.map((d, i) => {
              const x = xScale(d.group);
              const y = yScale(d.value);
              const barHeight = boundsHeight - y;
              return (
                <rect
                  key={i}
                  x={x}
                  y={y}
                  width={xScale.bandwidth()}
                  height={barHeight}
                  fill={fillColorScale(d.group)}
                  stroke="var(--hot-cinnamon)"
                  strokeWidth={1}
                  opacity={0.9}
                />
              );
            })}
          </g>

          {/* Value Labels */}
          <g>
            {sortedData.map((d, i) => {
              const x = xScale(d.group) + xScale.bandwidth() / 2;
              const y = yScale(d.value) - 5;
              return (
                <text
                  key={i}
                  x={x}
                  y={y}
                  textAnchor="middle"
                  alignmentBaseline="baseline"
                  fontSize={10}
                >
                  {d.value > 1000
                    ? `${(d.value / 1000).toFixed(1)}T`
                    : `${d.value}B`}
                </text>
              );
            })}
          </g>

          {/* Group Labels on X-axis */}
          <g>
            {sortedData.map((d, i) => {
              const x = xScale(d.group) + xScale.bandwidth() / 2;
              return (
                <text
                  key={i}
                  x={x}
                  y={boundsHeight + 15}
                  textAnchor="middle"
                  alignmentBaseline="hanging"
                  fontSize={10}
                >
                  {d.group}
                </text>
              );
            })}
          </g>
        </g>
      </svg>
    </div>
  );
};

export default Barplot;
