import katexify from '../../utils/katexify';
import './BRCA1.css';
import Accordion from '../UI/Accordion/Accordion';
import SideNote from '../UI/SideNote/SideNote';
import AnnotatedDnaGrid from '../DNASequence/AnnotatedDnaGrid';
import FacetedBeeswarmExample from './FacetedBeeswarm';
import SNVWindow from '../DNASequence/SNV';
import BRCA1Table from './BRCA1Table';

const BRCA1 = () => {
  return (
    <>
      <h1 className="body-subheader">BRCA1 Zero-Shot SNV Prediction</h1>
      <p className="body-text">
        The human <b>BRCA1</b> gene encodes a critical protein involved in
        repairing damaged DNA. Variants in this gene are strongly linked to an
        elevated risk of breast and ovarian cancers, making accurate variant
        classification clinically essential.
        <br />
        <br />
        <Accordion title="BRCA1" open={true}>
          <p className="body-text">
            <b>BRCA1</b> (Breast Cancer 1) is a critical tumor‑suppressor gene
            involved in DNA damage repair, and pathogenic variants in BRCA1
            dramatically increase the risk of hereditary breast and ovarian
            cancers.
          </p>
        </Accordion>
        Without any task-specific fine‑tuning and using only raw DNA sequences,
        Evo 2 can accurately predict the functional impact of clinically
        significant variants in the BRCA1 gene. In other words, Evo 2 can
        determine whether a particular single nucleotide variant of the BRCA1
        gene is likely to be harmful to the protein's function, and thus
        potentially increase the risk of cancer for the patient with the genetic
        variant.
      </p>

      <p className="body-text">
        <i>Single-nucleotide variants (SNVs)</i> are the simplest and most
        common type of genetic change, in which a single base pair in the DNA
        sequence is replaced by another. Depending on their molecular
        consequences, SNVs can be broadly classified as{' '}
        <i>loss‑of‑function (LOF)</i>—which abolish gene activity, often via
        premature stops or disrupted splicing—or as
        <i> functional/intermediate</i>—which preserve most of the protein's
        normal function, perhaps with modest perturbations.
      </p>
      <Accordion
        title="Single-Nucleotide Variants"
        open={true}
        className="chart-container-snv"
      >
        <p className="body-text">
          <b>Single-nucleotide variants (SNVs)</b> are the simplest and most
          common type of genetic change, in which a single base pair in the DNA
          sequence is replaced by another.
          <br />
          <br />
          Imagine, for example, a sample of a DNA sequence:
        </p>
        <AnnotatedDnaGrid
          margin={{ top: 5, bottom: 25, left: 45, right: 30 }}
          fill={false}
        />
        <p className="body-text">
          Perturbing the sequence at index 8 may result in a functional variant,
          or have little-to-no effect.
        </p>
        <AnnotatedDnaGrid
          margin={{ top: 45, bottom: 25, left: 45, right: 30 }}
          annotationCellIndex={8}
          annotationText="Swapping from a G to a T here may have little-to-no effect..."
          cellReplacement="T"
        />
        <p className="body-text">
          However, perturbing the sequence at index 20 may result in a LOF
          variant, disrupting the protein's function completely!
        </p>
        <AnnotatedDnaGrid
          margin={{ top: 45, bottom: 25, left: 45, right: 30 }}
          annotationCellIndex={20}
          annotationText="Swapping from a G to a C here may disrupt the protein's function completely!"
          cellReplacement="C"
        />
      </Accordion>
      <p className="body-text">
        To evaluate Evo 2's ability to distinguish loss‑of‑function (LOF) from
        functional or intermediate variants in BRCA1 without any additional
        training, the following analysis was conducted
        <SideNote>
          Multiple experiments were performed showcasing Evo 2's successful
          ability to predict the functional impact of genetic variants,
          including for BRCA2. For more details, see section 4.3.1 of the paper
          <sup>1</sup>.
        </SideNote>
        :
      </p>
      <p className="body-text">
        <b>1. Variant selection and labeling.</b>
        <br />
        All 3,893 single‑nucleotide variants (SNVs) in BRCA1 for which
        experimental functional scores and classifications were available were
        gathered
        <a id="citation-6" href="#ref-6" className="citation-link">
          <sup>6</sup>
        </a>
        . SNVs labeled as “LOF” were classified as loss of function variants (𝑁
        = 823), while SNVs labeled as “FUNC” or “INT” were labeled as
        functional/intermediate variants (𝑁 = 3,070). These function scores
        reflect the extent by which the genetic variant has disrupted the
        protein's function, with lower scores indicating greater disruption.
      </p>
      <p className="body-text">
        <b>2. Sequence extraction.</b>
        <br />
        For each SNV, an 8,192-base-pair genomic window centered on the variant
        site was extracted from the human reference genome, ensuring consistency
        with previous studies.
      </p>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <SNVWindow
          margin={{ top: 45, bottom: 25, left: 45, right: 30 }}
          annotationCellIndex={20}
          annotationText="8,192-base-pair genomic window centered over the variant site"
          cellReplacement="C"
        />
      </div>

      <p className="body-text">
        <b>3. Zero‑shot scoring.</b>
        <br />
        Without any additional training, Evo 2 evaluated each SNV by calculating
        log-likelihood scores for both the original (<i>wild-type</i>) sequences
        and the variant sequences. Higher likelihood scores indicated sequences
        that the model recognized as more "natural" or typical, based on the
        patterns and underlying genetic relationships learned during training.
        To quantify each variant's impact, a delta likelihood score was
        calculated:{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(
              '\\Delta = \\log P(\\text{variant}) - \\log P(\\text{reference})',
              true
            ),
          }}
        />
        This delta score captures how much the variant changes the sequence
        likelihood relative to the <i>wild-type</i>. The key intuition is that
        disruptive variants should have lower likelihood scores than their
        reference sequences, resulting in negative delta scores. The more
        negative the delta, the more the model considers the variant to be
        "unnatural" or disruptive to normal genomic patterns.
        <br />
        <br />A sample of the zero‑shot likelihood distributions for LOF versus
        functional/intermediate variants is shown below
        <SideNote>
          Check out the{' '}
          <a
            href="https://github.com/NVIDIA/bionemo-framework/blob/main/sub-packages/bionemo-evo2/examples/zeroshot_brca1.ipynb"
            target="_blank"
          >
            notebook
          </a>{' '}
          to recreate this experiment for yourself!
        </SideNote>
        . Note that the SNV data is for the BRCA1 window in GRCh37/hg19, which
        runs from 41196312–41322262 bp:
      </p>

      <FacetedBeeswarmExample />

      <p className="body-text">
        The results show a clear separation between classes (two-sided Wilcoxon
        rank‑sum test), with a record-setting AUROC score of 0.87. Evo 2's raw
        sequence model thus provides meaningful functional signals
        out-of-the-box.
      </p>
      <BRCA1Table />
    </>
  );
};

export default BRCA1;
