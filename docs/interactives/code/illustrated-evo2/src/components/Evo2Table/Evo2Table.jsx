import React from 'react';
import './Evo2Table.css';

const Evo2Table = () => {
  // You can set which row to highlight (for example, row index 5 for the Total Tokens row)
  const highlightRowIndex = 5;

  const rows = [
    {
      id: 'parameters',
      label: 'Parameters',
      evo40B: '40.3 B',
      evo7B: '6.5 B',
      evo1B: '1.1 B',
    },
    {
      id: 'layers',
      label: 'Total Layers',
      evo40B: '50',
      evo7B: '32',
      evo1B: '25',
    },
    {
      id: 'hidden',
      label: 'Hidden Size',
      evo40B: '8,192',
      evo7B: '4,096',
      evo1B: '1,920',
    },
    {
      id: 'ffn',
      label: 'FFN Size',
      evo40B: '22,528',
      evo7B: '11,264',
      evo1B: '5,120',
    },
    { id: 'heads', label: 'Num Heads', evo40B: '64', evo7B: '32', evo1B: '15' },
    {
      id: 'tokens',
      label: 'Total Tokens',
      evo40B: '9.3 T',
      evo7B: '2.4 T',
      evo1B: '1 T',
      isBold: false,
    },
  ];

  return (
    <div className="table-container">
      <table>
        <thead>
          <tr>
            <th>Evo 2 Spec</th>
            <th>1B</th>
            <th>7B</th>
            <th>40B</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={row.id}
              className={index === highlightRowIndex ? 'highlighted' : ''}
            >
              <td>{row.isBold ? <strong>{row.label}</strong> : row.label}</td>
              <td>{row.isBold ? <strong>{row.evo1B}</strong> : row.evo1B}</td>
              <td>{row.isBold ? <strong>{row.evo7B}</strong> : row.evo7B}</td>
              <td>{row.isBold ? <strong>{row.evo40B}</strong> : row.evo40B}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Evo2Table;
