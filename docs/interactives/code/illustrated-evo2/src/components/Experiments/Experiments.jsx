import './Experiments.css';
import Accordion from '../UI/Accordion/Accordion';

const Experiments = () => {
  return (
    <>
      <h1 className="body-header">Experiments</h1>
      <p className="body-text">
        To showcase the capabilities of Evo 2, several experiments were
        conducted highlighting its versatility and generalist approach to
        genomic modeling. Below, we will highlight two key areas of experiments:
        Variant Effect Prediction and Genome Generation.
        <br />
        <br />
        <Accordion
          title="Variant Effect Prediction & Genome Generation"
          open={true}
        >
          <p className="body-text">
            <b>Variant Effect Prediction (VEP)</b>: refers to the task of
            predicting the functional impact of a genetic variant on a gene.
          </p>
          <p className="body-text">
            <b>Genome Generation</b>: refers to the task of generating a new
            genome sequence.
          </p>
        </Accordion>
        We'll start with Variant Effect Prediction, covering Evo 2's application
        to the BRCA1 gene.
      </p>
    </>
  );
};

export default Experiments;
