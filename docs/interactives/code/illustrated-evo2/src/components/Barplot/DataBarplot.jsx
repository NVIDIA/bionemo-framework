import { data, ncbi_genomes } from './data';
import Barplot from './Barplot';
import { VerticalBarplot } from './VerticalBarplot';
import './DataBarplot.css';

export const DataBarplot = () => {
  return (
    <div className="chart-container-flex">
      <VerticalBarplot data={data} width={420} height={140} />
      <Barplot data={ncbi_genomes} width={300} height={140} />
    </div>
  );
};
