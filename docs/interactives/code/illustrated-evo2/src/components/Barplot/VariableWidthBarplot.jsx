import { useMemo } from 'react';
import * as d3 from 'd3';

const MARGIN = { top: 40, right: 120, bottom: 100, left: 60 };
const GAP_WITHIN = 120; // Gap between pretraining and midtraining
const GAP_BETWEEN = 800; // Gap between 7B and 40B models

// Training dataset composition data
const COMPOSITION = {
  pretraining: {
    Organelles: 2,
    'Eukaryotic promoter + exons + splice sites': 22,
    'Eukaryotic mRNAs': 10,
    'Eukaryotic 5kb windows': 5,
    ncRNA: 5,
    EFDnew: 3,
    GTDB: 25,
    Metagenomics: 18,
    'IMG/VR': 10,
  },
  midtraining: {
    Organelles: 2,
    'Eukaryotic promoter + exons + splice sites': 15,
    'Eukaryotic mRNAs': 15,
    'Eukaryotic 5kb windows': 10,
    ncRNA: 5,
    EFDnew: 3,
    GTDB: 10,
    Metagenomics: 10,
    'IMG/VR': 5,
    Animalia: 15,
    Plantae: 5,
    Fungi: 3,
    Protista: 2,
  },
};

// Token volumes data
const TOKENS = {
  'Evo 2 7B': {
    pretraining: 2000,
    midtraining: 500,
  },
  'Evo 2 40B': {
    pretraining: 8500,
    midtraining: 1000,
  },
};

// Color mapping
const COLORS = {
  // NCBI Genomes - orange/red scale
  Protista: '#650b0b',
  Fungi: '#df6500',
  Plantae: '#ef9100',
  Animalia: '#f9c500',
  // Prokaryotic - green scale
  'IMG/VR': '#3f8500',
  Metagenomics: '#bff230',
  GTDB: '#76b900',
  // Eukaryotic genic - blue scale
  EFDnew: 'var(--mine-shaft)',
  ncRNA: 'var(--dove-gray)',
  'Eukaryotic 5kb windows': 'var(--gray)',
  'Eukaryotic mRNAs': 'var(--silver)',
  'Eukaryotic promoter + exons + splice sites': 'var(--silver-chalice)',
  // Organelles - purple scale
  Organelles: 'var(--cerise)',
};

const CATEGORY_GROUPS = [
  {
    name: 'NCBI Genomes',
    color: '#f9c500',
    items: ['Animalia', 'Plantae', 'Fungi', 'Protista'],
  },
  {
    name: 'Prokaryotic',
    color: '#76b900',
    items: ['GTDB', 'Metagenomics', 'IMG/VR'],
  },
  {
    name: 'Eukaryotic genic',
    color: '#0074df',
    items: [
      'Eukaryotic promoter + exons + splice sites',
      'Eukaryotic mRNAs',
      'Eukaryotic 5kb windows',
      'ncRNA',
      'EFDnew',
    ],
  },
  { name: 'Organelles', color: '#4d1368', items: ['Organelles'] },
];

export const VariableWidthBarplot = ({ width = 870, height = 450 }) => {
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  // Prepare bar data with positions and segments
  const barsData = useMemo(() => {
    const bars = [];
    let xPosition = 0;

    Object.entries(TOKENS).forEach(([model, phases], modelIndex) => {
      ['pretraining', 'midtraining'].forEach((phase, phaseIndex) => {
        // Add spacing
        if (modelIndex === 0 && phaseIndex === 1) xPosition += GAP_WITHIN;
        if (modelIndex === 1 && phaseIndex === 0) xPosition += GAP_BETWEEN;
        if (modelIndex === 1 && phaseIndex === 1) xPosition += GAP_WITHIN;

        // Build segments for this bar
        const segments = [];
        let yPosition = 0;

        Object.entries(COMPOSITION[phase]).forEach(([category, percentage]) => {
          segments.push({
            category,
            y0: yPosition,
            y1: yPosition + percentage,
            percentage,
            color: COLORS[category],
          });
          yPosition += percentage;
        });

        bars.push({
          model,
          phase,
          width: phases[phase],
          x: xPosition,
          segments,
          modelIndex,
          phaseIndex,
        });

        xPosition += phases[phase];
      });
    });

    return bars;
  }, []);

  // Calculate scales
  const totalWidth = useMemo(() => {
    return d3.sum(barsData, d => d.width) + GAP_WITHIN * 2 + GAP_BETWEEN;
  }, [barsData]);

  const xScale = useMemo(() => {
    return d3.scaleLinear().domain([0, totalWidth]).range([0, boundsWidth]);
  }, [totalWidth, boundsWidth]);

  const yScale = useMemo(() => {
    return d3.scaleLinear().domain([0, 100]).range([boundsHeight, 0]);
  }, [boundsHeight]);

  // Render segments for each bar
  const barGroups = barsData.map((bar, i) => (
    <g key={i} transform={`translate(${xScale(bar.x)}, 0)`}>
      {/* Bar segments */}
      {bar.segments.map((segment, j) => (
        <rect
          key={j}
          x={0}
          y={yScale(segment.y1)}
          width={xScale(bar.width) - xScale(0)}
          height={yScale(segment.y0) - yScale(segment.y1)}
          fill={segment.color}
          stroke="var(--nvgreen)"
          strokeWidth={0}
          opacity={0.99}
        >
          <title>
            {segment.category}: {segment.percentage}% of {bar.phase}
            {'\n'}Model: {bar.model}
            {'\n'}Volume: {Math.round((bar.width * segment.percentage) / 100)}B
            nucleotides
          </title>
        </rect>
      ))}

      {/* Phase label */}
      <text
        x={xScale(bar.width) / 2 - xScale(0) / 2}
        y={boundsHeight + 20}
        textAnchor="middle"
        fontSize={10}
        fill="#666"
      >
        {bar.phase.charAt(0).toUpperCase() + bar.phase.slice(1)}
      </text>

      {/* Volume label */}
      <text
        x={xScale(bar.width) / 2 - xScale(0) / 2}
        y={boundsHeight + 35}
        textAnchor="middle"
        fontSize={10}
        fill="#888"
      >
        {bar.width.toLocaleString()}B tokens
      </text>
    </g>
  ));

  // Model labels
  const modelLabels = barsData
    .filter(d => d.phaseIndex === 0)
    .map((bar, i) => (
      <text
        key={i}
        x={xScale(bar.x) + xScale(bar.width) / 2}
        y={-10}
        textAnchor="middle"
        fontSize={14}
        fontWeight="bold"
      >
        {bar.model}
      </text>
    ));

  // Model divider
  const model1End = xScale(barsData[1].x + barsData[1].width);
  const model2Start = xScale(barsData[2].x);
  const dividerX = (model1End + model2Start) / 2;

  // Y-axis ticks
  const yAxisTicks = yScale.ticks(5).map((tick, i) => (
    <g key={i}>
      <line x1={-5} x2={0} y1={yScale(tick)} y2={yScale(tick)} stroke="black" />
      <text
        x={-10}
        y={yScale(tick)}
        textAnchor="end"
        alignmentBaseline="middle"
        fontSize={12}
      >
        {tick}
      </text>
    </g>
  ));

  // Legend with proper spacing
  const legendItems = Object.entries(COLORS).map((d, i) => ({
    category: d[0],
    color: d[1],
  }));

  const legend = (
    <g transform={`translate(${boundsWidth + 20}, 20)`}>
      {CATEGORY_GROUPS.map((group, groupIndex) => {
        const groupItems = legendItems.filter(item =>
          group.items.includes(item.category)
        );

        // Calculate cumulative Y position with proper spacing
        let cumulativeY = 0;
        for (let i = 0; i < groupIndex; i++) {
          const prevGroupItems = legendItems.filter(item =>
            CATEGORY_GROUPS[i].items.includes(item.category)
          );
          cumulativeY += prevGroupItems.length * 15 + 35; // 20px per item + 35px group spacing
        }

        return (
          <g key={groupIndex} transform={`translate(${0}, -7.5)`}>
            <text
              x={-5}
              y={cumulativeY - 7.5}
              fontSize={9}
              fontWeight="bold"
              fill={'var(--mine-shaft)'}
            >
              {group.name}
            </text>
            {groupItems.map((item, i) => (
              <g
                key={i}
                transform={`translate(0, ${cumulativeY + 2 + i * 15})`}
              >
                <rect width={9} height={9} fill={item.color} />
                <text x={13.5} y={8} fontSize={8}>
                  {item.category.length > 30
                    ? item.category.substring(0, 27) + '...'
                    : item.category}
                </text>
              </g>
            ))}
          </g>
        );
      })}
    </g>
  );

  return (
    <div>
      <svg width={width} height={height}>
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* Y-axis */}
          <line x1={0} y1={0} x2={0} y2={boundsHeight} stroke="black" />
          {yAxisTicks}

          {/* X-axis label */}
          <text
            x={boundsWidth / 2}
            y={boundsHeight + 70}
            textAnchor="middle"
            fontSize={9}
            fill="var(--dove-gray)"
          >
            Token Volume (width proportional to billions of nucleotides). In all
            models, the majority of data (and FLOPS) is dedicated to
            pretraining.
          </text>

          {/* Bars */}
          {barGroups}

          {/* Model labels */}
          {modelLabels}

          {/* Model divider */}
          <line
            x1={dividerX}
            y1={-20}
            x2={dividerX}
            y2={boundsHeight + 40}
            stroke="#ddd"
            strokeWidth={0}
            strokeDasharray="5,5"
          />

          {/* Legend */}
          {legend}
        </g>
      </svg>
    </div>
  );
};

export default VariableWidthBarplot;
