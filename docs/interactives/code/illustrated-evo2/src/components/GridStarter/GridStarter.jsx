import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const GridStarter = () => {
  const svgRef = useRef();
  const width = 1000;
  const height = 800;
  const MARGIN = { top: 100, right: 30, bottom: 150, left: 50 };
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  const gridOpacity = 1;
  const maxX = 16;
  const maxY = 16;

  const xScale = d3.scaleLinear().domain([0, maxX]).range([0, boundsWidth]);

  const yScale = d3.scaleLinear().domain([0, maxY]).range([boundsHeight, 0]);

  const xGridLines = () => {
    return xScale.ticks(16).map((_, i) => {
      return (
        <line
          x1={xScale(i)}
          x2={xScale(i)}
          y1={yScale(0)}
          y2={yScale(maxY)}
          stroke="#e0e0e0"
          strokeDasharray="2,2"
          opacity={gridOpacity}
        />
      );
    });
  };

  const yGridLines = () => {
    return yScale.ticks(12).map((_, i) => {
      return (
        <line
          x1={xScale(0)}
          x2={xScale(maxX)}
          y1={yScale(i)}
          y2={yScale(i)}
          stroke="#e0e0e0"
          strokeDasharray="2,2"
          opacity={gridOpacity}
        />
      );
    });
  };

  return (
    <div style={{ margin: '0 auto', width: '900px' }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="h2"
        style={{
          fontFamily:
            'NVIDIA Sans, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        }}
      >
        <g
          width={boundsWidth}
          height={boundsHeight}
          transform={`translate(${[MARGIN.left, MARGIN.top].join(',')})`}
          overflow={'visible'}
        >
          {/* {drawGrid()} */}
          {/* {drawCircle()} */}
          {xGridLines()}
          {yGridLines()}
        </g>
      </svg>
    </div>
  );
};
export default GridStarter;
