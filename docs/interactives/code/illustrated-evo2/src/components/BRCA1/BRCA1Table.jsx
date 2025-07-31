import React from 'react';
import './BRCA1Table.css';

const BRCA1Table = () => {
  const rows = [
    { modelSize: 'Evo 2 1B', auroc: 0.74 },
    { modelSize: 'Evo 2 7B', auroc: 0.87 },
  ];

  return (
    <div className="table-container">
      <table className="evo2-table">
        <thead>
          <tr>
            <th>Model Size</th>
            <th>AUROC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.modelSize}>
              <td>{row.modelSize}</td>
              <td>{row.auroc.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default BRCA1Table;
