import React, { useMemo } from 'react';
import * as d3 from 'd3';
import { genomeData } from './genomeData';

export const GenomeGenerationTable = () => {
  // 1) Sort the data once
  const sortedData = useMemo(
    () => [...genomeData].sort((a, b) => b.value - a.value),
    []
  );

  // 2) Build a linear color scale from min→max mapped to our CSS vars
  const colorScale = useMemo(() => {
    // get the numeric extent [min, max]
    const [min, max] = d3.extent(sortedData, d => d.value);

    // pull the actual color values for our CSS variables
    const rootStyles = getComputedStyle(document.documentElement);
    const startColor = rootStyles.getPropertyValue('--jonquil').trim();
    const endColor = rootStyles.getPropertyValue('--nvgreen').trim();

    // build and return the scale
    return d3.scaleLinear().domain([min, max]).range([startColor, endColor]);
  }, [sortedData]);

  return (
    <div className="table-container">
      <table className="genome-table">
        <thead>
          <tr>
            <th>Sequence ID</th>
            <th>Percentage</th>
            <th>Nearest Species</th>
          </tr>
        </thead>
        <tbody>
          {sortedData.map(d => (
            <tr key={d.group}>
              <td>{d.group}</td>
              <td
                style={{
                  backgroundColor: colorScale(d.value),
                  transition: 'background-color 0.3s',
                  fontWeight: 'bold',
                }}
              >
                {d.value.toFixed(2)}%
              </td>
              <td>{d.nearest}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default GenomeGenerationTable;
