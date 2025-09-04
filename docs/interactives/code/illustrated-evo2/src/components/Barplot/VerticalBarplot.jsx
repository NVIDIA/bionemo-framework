import { useMemo } from 'react';
import * as d3 from 'd3';

const MARGIN = { top: 0, right: 45, bottom: 10, left: 150 };
const BAR_PADDING = 0.2;

// Color palette using CSS variables
const FILL_COLORS = [
  'var(--nvgreen)', // #76b900
  'var(--seance)', // #9525c6
  'var(--carrot-orange)', // #ef9100
  'var(--curious-blue)', // #0074df
  'var(--brick-red)', // #e52020
  'var(--cerise)', // #d2308e
  'var(--mountain-meadow)', // #1dbba4
  'var(--amber)', // #f9c500
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

export const VerticalBarplot = ({ width, height, data }) => {
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  const sortedData = data.sort((a, b) => b.value - a.value);
  const maxValue = sortedData[0].value;

  const yScale = useMemo(() => {
    return d3
      .scaleBand()
      .domain(sortedData.map(d => d.group))
      .range([0, boundsHeight])
      .padding(BAR_PADDING);
  }, [data, height]);

  const xScale = useMemo(() => {
    return d3.scaleLinear().domain([0, maxValue]).range([0, boundsWidth]);
  }, [data, width]);

  const fillColorScale = d3
    .scaleOrdinal()
    .domain(data.map(d => d.group))
    .range(FILL_COLORS);

  const strokeColorScale = d3
    .scaleOrdinal()
    .domain(data.map(d => d.group))
    .range(STROKE_COLORS);

  const rectangles = sortedData.map((d, i) => {
    return (
      <rect
        key={i}
        y={yScale(d.group)}
        height={yScale.bandwidth()}
        x={0}
        width={xScale(d.value)}
        // fill={fillColorScale(d.group)}
        // stroke={strokeColorScale(d.group)}
        fill="var(--inch-worm)"
        stroke="var(--deep-fir)"
        strokeWidth={1}
        opacity={0.9}
      />
    );
  });

  const labels = sortedData.map((d, i) => {
    const y = yScale(d.group);
    if (y === undefined) return null;

    return (
      <g key={i}>
        <text
          x={xScale(d.value) + 10}
          y={y + yScale.bandwidth() / 2}
          textAnchor="start"
          alignmentBaseline="central"
          fontSize={10}
        >
          {d.value > 1000 ? `${(d.value / 1000).toFixed(1)}T` : `${d.value}B`}
        </text>
        <text
          x={-10}
          y={y + yScale.bandwidth() / 2}
          textAnchor="end"
          alignmentBaseline="central"
          fontSize={10}
        >
          {d.group}
        </text>
      </g>
    );
  });

  const grid = xScale
    .ticks(5)
    .slice(1)
    .map((value, i) => (
      <g key={i}>
        <line
          x1={xScale(value)}
          x2={xScale(value)}
          y1={0}
          y2={boundsHeight}
          stroke="#808080"
          opacity={0.2}
        />
        <text
          x={xScale(value)}
          y={boundsHeight + 10}
          textAnchor="middle"
          alignmentBaseline="central"
          fontSize={9}
          opacity={0.8}
        >
          {value > 1000 ? `${(value / 1000).toFixed(1)}T` : `${value}B`}
        </text>
      </g>
    ));

  return (
    <div className="barplot-container">
      <p className="loss-chart-title">
        OpenGenome2 Dataset: Nucleotides Count{' '}
      </p>
      <svg width={width} height={height}>
        <g
          width={boundsWidth}
          height={boundsHeight}
          transform={`translate(${[MARGIN.left, MARGIN.top].join(',')})`}
        >
          {/* <g>{grid}</g> */}
          <g>{rectangles}</g>
          <g>{labels}</g>
        </g>
      </svg>
    </div>
  );
};
