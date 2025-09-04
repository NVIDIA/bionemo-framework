// DnaSequence.jsx
import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const DnaSequence = ({
  width = 800,
  height = 40,
  margin = { top: 30, bottom: 30 },
}) => {
  const ref = useRef();

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    // Define your segments and their relative lengths & colors
    const segments = [
      { length: 10, color: '#7285b7', type: 'selfish element' },
      { length: 15, color: '#283250', type: 'selfish element' },
      { length: 10, color: '#8da4d4', type: 'selfish element' },
      { length: 10, color: '#bf7f2a', type: 'essential gene' },
      { length: 10, color: '#8da4d4', type: 'selfish element' },
      { length: 80, color: '#d0dbe8', type: 'repeat expansion' },
      { length: 10, color: '#8da4d4', type: 'selfish element' },
      { length: 15, color: '#283250', type: 'selfish element' },
      { length: 10, color: '#7285b7', type: 'selfish element' },
    ];

    // Compute scales
    const total = d3.sum(segments, d => d.length);
    const xScale = d3.scaleLinear().domain([0, total]).range([0, width]);

    // Draw the bars
    let cursor = 0;
    const g = svg.append('g').attr('transform', `translate(0, ${margin.top})`);

    segments.forEach(seg => {
      g.append('rect')
        .attr('x', xScale(cursor))
        .attr('y', 0)
        .attr('width', xScale(seg.length))
        .attr('height', height)
        .attr('fill', seg.color);
      cursor += seg.length;
    });

    // Helper to place a centered label over a span of segments
    function addLabel(startIdx, endIdx, text, y) {
      const x0 = d3.sum(segments.slice(0, startIdx), d => d.length);
      const x1 = d3.sum(segments.slice(0, endIdx + 1), d => d.length);
      const xm = xScale((x0 + x1) / 2);
      svg
        .append('text')
        .attr('x', xm)
        .attr('y', y)
        .attr('text-anchor', 'middle')
        .attr('font-family', 'sans-serif')
        .attr('font-size', 12)
        .text(text);
    }

    // Top labels
    addLabel(0, 2, 'Selfish element', margin.top - 8);
    addLabel(5, 5, 'Repeat expansion', margin.top - 8);
    addLabel(6, 8, 'Selfish element', margin.top - 8);
    // Bottom label
    addLabel(3, 3, 'Essential gene', margin.top + height + 15);
  }, [width, height, margin]);

  return (
    <svg ref={ref} width={width} height={height + margin.top + margin.bottom} />
  );
};

export default DnaSequence;
