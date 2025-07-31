import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const RepetitionChart = ({
  sequence = 'ACGTAGCTAGCTTACGATCGATCGATCGGATCGATCGATCGGAT',
  repeatPattern = '10-15,20-30',
  width = 600,
  height = 45,
}) => {
  // Process the repeat regions from string input
  const processRepeatPattern = pattern => {
    const repeatSet = new Set();

    // Split by commas to handle multiple ranges
    const ranges = pattern.split(',').map(r => r.trim());

    ranges.forEach(range => {
      if (range.includes('-')) {
        // Handle ranges like "10-15"
        const [start, end] = range
          .split('-')
          .map(num => parseInt(num.trim(), 10));
        if (!isNaN(start) && !isNaN(end)) {
          for (let i = start; i <= end; i++) {
            repeatSet.add(i);
          }
        }
      } else {
        // Handle single numbers
        const idx = parseInt(range.trim(), 10);
        if (!isNaN(idx)) {
          repeatSet.add(idx);
        }
      }
    });

    return repeatSet;
  };

  const repeatSet = processRepeatPattern(repeatPattern);

  // Build data array of { idx, base, weight }
  const data = sequence.split('').map((base, i) => ({
    idx: i,
    base,
    weight: repeatSet.has(i) ? 0.1 : 1,
  }));

  const svgRef = useRef();

  useEffect(() => {
    const svg = d3.select(svgRef.current);

    // Clear previous chart
    svg.selectAll('*').remove();

    // Set dimensions
    svg.attr('width', width).attr('height', height + 30);

    // X & Y scales
    const x = d3
      .scaleBand()
      .domain(data.map(d => d.idx))
      .range([0, width])
      .padding(0.25);

    const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

    // Bars
    svg
      .append('g')
      .selectAll('rect')
      .data(data)
      .join('rect')
      .attr('x', d => x(d.idx))
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.weight))
      .attr('height', d => height - y(d.weight))
      .attr(
        'fill',
        d =>
          d.weight === 1
            ? 'var(--inch-worm)' // non-repeat color (inch-worm)
            : 'var(--pippin)' // repeat color (gorse)
      )
      .attr(
        'stroke',
        d =>
          d.weight === 1
            ? 'var(--deep-fir)' // non-repeat stroke (deep-fir)
            : 'var(--geraldine)' // repeat stroke (hot-cinnamon)
      )
      .attr('stroke-width', 1.2);

    // X-axis: one tick & label per base
    svg
      .append('g')
      .attr('transform', `translate(0,${height})`)
      .call(
        d3
          .axisBottom(x)
          .tickValues(data.map(d => d.idx))
          .tickFormat(i => data[i].base)
          .tickSize(0)
      )
      .selectAll('text')
      .attr('dy', '1em')
      .style('font-size', '10px');
  }, [data, width, height]);

  return (
    <div className="loss-container">
      <svg ref={svgRef} className="repetition-chart" />
    </div>
  );
};

export default RepetitionChart;
